"""Prior dictionary for BNS systems using chirp mass and tidal deformability.

This module provides a specialized prior dictionary for binary neutron star (BNS) systems
that uses chirp mass and tidal deformability (lambda_tilde) as primary parameters.
"""

# Standard library imports
import numpy as np
import wcosmo
import pandas as pd
import matplotlib.pyplot as plt

# Third-party imports
from scipy.interpolate import CloughTocher2DInterpolator

# Bilby imports
from bilby.core.prior import (
    Prior, 
    Uniform, 
    Interped,
    ConditionalInterped,
    PriorDict,
    ConditionalLogUniform,
    LogUniform,
    TruncatedGaussian,
    LogNormal,
    ConditionalUniform,
    Constraint,
    ConditionalPriorDict,
    Cosine,
    Sine
)
from bilby.gw.prior import BNSPriorDict, UniformInComponentsChirpMass
from bilby.gw.conversion import (
    lambda_tilde_to_lambda_1_lambda_2,
    component_masses_to_symmetric_mass_ratio,
    luminosity_distance_to_redshift,
    generate_mass_parameters
)
from bilby.core.utils import logger


def lambda_tilde_to_lambda_1_lambda_2_NSBH(lambda_tilde, mass_1, mass_2):
    """Convert from dominant tidal term to individual tidal parameters for NSBH.

    Converts the dominant tidal term to individual tidal parameters assuming lambda_1 = 0
    for neutron star-black hole systems.

    Args:
        lambda_tilde: Dominant tidal term.
        mass_1: Mass of more massive neutron star.
        mass_2: Mass of less massive neutron star.

    Returns:
        tuple: (lambda_1, lambda_2) where:
            lambda_1: Tidal parameter of more massive neutron star.
            lambda_2: Tidal parameter of less massive neutron star.
    """
    eta = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)
    coefficient = ((1 + 7 * eta - 31 * eta**2) - 
                  (1 - 4 * eta)**0.5 * (1 + 9 * eta - 11 * eta**2))
    
    lambda_1 = 0.0
    lambda_2 = 13 / 8 * lambda_tilde / coefficient
    
    return lambda_1, lambda_2


class BNSPriorDict_chirpmass_lambda_tilde(BNSPriorDict):
    """Prior dictionary for BNS systems parameterized by chirp mass and tidal deformability.
    
    This class extends BNSPriorDict to handle priors on chirp mass and lambda_tilde,
    either using default uniform priors or interpolated from provided data.

    Attributes:
        MCL_filename: Path to the file containing prior distribution data.
        f_interp: Interpolator for the probability distribution.
        tides_sort: Sorted array of unique tidal deformability values.
    """

    def __init__(self, MCL_filename=None, dictionary=None, filename=None,
                 conversion_function=None):
        """Initialize the prior dictionary.

        Args:
            MCL_filename: Path to the file containing prior distribution data.
            dictionary: Dictionary of prior objects.
            filename: Path to a file containing prior specifications.
            conversion_function: Function to convert between parameters.
        """
        self.MCL_filename = MCL_filename
        super().__init__(
            dictionary=dictionary,
            filename=filename,
            conversion_function=conversion_function
        )

        self._remove_unnecessary_priors()
        
        if self.MCL_filename is None:
            logger.warning('No prior distribution filename given, defaulting to uniform priors.')
            self._setup_default_priors()
        else:
            logger.info('Interpolating chirp_mass_source and lambda_tilde prior from file.')
            self._setup_interpolated_priors(self.MCL_filename)

    def _remove_unnecessary_priors(self):
        """Remove priors that will be replaced with new ones."""
        for param in ['lambda_1', 'lambda_2', 'chi_1', 'chi_2', 'chirp_mass']:
            self.pop(param, None)

    def _setup_interpolated_priors(self, filename):
        """Set up interpolated priors from data file.

        Args:
            filename: Path to the file containing prior distribution data.
        """
        df = pd.read_csv(filename, delim_whitespace=True)
        mass = np.array(df['mass'].tolist())
        tides = np.array(df['tides'].tolist())
        prob = np.array(df['prob'].tolist())
        
        # Normalize probability
        prob = prob / np.sum(prob)
        
        # Create points for interpolation
        points = np.column_stack((mass, tides))
        mass_sort = np.sort(np.unique(mass))
        self.tides_sort = np.sort(np.unique(tides))

        # Set up interpolator
        self.f_interp = CloughTocher2DInterpolator(points, prob)

        # Reshape data for marginalization
        data = df.to_numpy()
        grid_x_size = mass_sort.shape[0]
        grid_y_size = self.tides_sort.shape[0]
        reshaped_data = data[:, 2].reshape((grid_x_size, grid_y_size))

        # Marginalize over tidal axis for chirp mass prior
        chirp_mass_source = np.sum(reshaped_data, axis=0)
        chirp_mass_source = chirp_mass_source / np.sum(chirp_mass_source)

        # Set up priors
        self['chirp_mass_source'] = Interped(xx=mass_sort, yy=chirp_mass_source)
        self['lambda_tilde'] = ConditionalInterped(
            xx=self.tides_sort,
            yy=chirp_mass_source,
            condition_func=self.conditional_func_y
        )

    def _setup_default_priors(self):
        """Set up default uniform priors when no data file is provided."""
        self['chirp_mass_source'] = UniformInComponentsChirpMass(
            minimum=0.1,
            maximum=2.0,
            name='chirp_mass',
            latex_label='$\\mathcal{M}$'
        )
        
        self['lambda_tilde'] = Uniform(
            minimum=0.0,
            maximum=1000.0,
            name='lambda_tilde',
            latex_label=r'$\tilde{\Lambda}$'
        )

    def conditional_func_y(self, reference_parameters, chirp_mass_source):
        """Condition function for lambda_tilde prior dependent on chirp_mass_source.

        Args:
            reference_parameters: Dictionary containing original xx and yy values.
            chirp_mass_source: Value(s) drawn from the chirp_mass_source prior.

        Returns:
            dict: Dictionary with 'yy' containing interpolated lambda_tilde values.
        """
        tides_sort = self.tides_sort 
        
        if isinstance(chirp_mass_source, pd.core.series.Series):
            # TODO: Fix this hack for probability calculation at end of sampling
            yy = reference_parameters['yy']
        else:
            y0_tmp = np.tile(chirp_mass_source, len(tides_sort))
            # TODO: Handle negative values, NaN, etc. from interpolation
            yy = self.f_interp(y0_tmp, tides_sort)

        return dict(yy=yy)




def set_backend(backend):
    import wcosmo
    from importlib import import_module
    np_modules = dict(
        numpy="numpy",
        jax="jax.numpy",
        cupy="cupy",
    )
    linalg_modules = dict(
        numpy="scipy.linalg",
        jax="jax.scipy.linalg",
        cupy="cupyx.scipy.linalg",
    )
    setattr(wcosmo.wcosmo, "xp", import_module(np_modules[backend]))
    setattr(wcosmo.utils, "xp", import_module(np_modules[backend]))
    toeplitz = getattr(import_module(linalg_modules[backend]), "toeplitz")
    setattr(wcosmo.utils, "toeplitz", toeplitz)

def convert_to_lal_binary_neutron_star_parameters_mchirp(parameters):
    """
    Convert parameters for BNS systems parameterized by chirp mass.

    Required parameters are:
    Mass: chirp_mass(_source), mass_ratio
    Spin: a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl
    Extrinsic: luminosity_distance, theta_jn, phase, ra, dec, geocent_time, psi
    Tidal: lambda_tilde

    Parameters
    ==========
    parameters: dict
        Dictionary of parameter values to convert
    
    Returns
    =======
    converted_parameters: dict
        Dictionary with all required parameters
    added_keys: list
        Keys which were added during conversion
    """
    from bilby.gw.conversion import generate_mass_parameters, generate_tidal_parameters
    import numpy as np
    from bilby.core.utils import logger
    from wcosmo.astropy import Planck15  # Import Planck15 cosmology

    set_backend("numpy")
    
    converted_parameters = parameters.copy()
    #print(converted_parameters)
    original_keys = list(converted_parameters.keys())

    # Use planck15 cosmology from wcosmo
    cosmology = Planck15
    #print(cosmology)
    # Handle distance/redshift conversions
    if 'luminosity_distance' not in original_keys:
        if 'redshift' in converted_parameters:
            converted_parameters['luminosity_distance'] = \
                cosmology.luminosity_distance(converted_parameters['redshift'])
        elif 'comoving_distance' in converted_parameters:
            z = cosmology.z_at_comoving_distance(converted_parameters['comoving_distance'])
            converted_parameters['luminosity_distance'] = \
                cosmology.luminosity_distance(z)

    # Convert source frame parameters
    for key in original_keys:
        if key.endswith('_source'):
            if 'redshift' not in converted_parameters:
                converted_parameters['redshift'] = \
                    luminosity_distance_to_redshift(
                        converted_parameters['luminosity_distance']
                    )
            converted_parameters[key[:-7]] = converted_parameters[key] * (
                1 + converted_parameters['redshift'])

    # Generate all mass parameters from chirp mass and mass ratio
    converted_parameters = generate_mass_parameters(converted_parameters)

    # Handle spin conversions
    for idx in ['1', '2']:
        key = f'chi_{idx}'
        if key in original_keys:
            if f"chi_{idx}_in_plane" in original_keys:
                converted_parameters[f"a_{idx}"] = (
                    converted_parameters[f"chi_{idx}"] ** 2
                    + converted_parameters[f"chi_{idx}_in_plane"] ** 2
                ) ** 0.5
                converted_parameters[f"cos_tilt_{idx}"] = (
                    converted_parameters[f"chi_{idx}"]
                    / converted_parameters[f"a_{idx}"]
                )
            elif f"a_{idx}" not in original_keys:
                converted_parameters[f'a_{idx}'] = abs(
                    converted_parameters[key])
                converted_parameters[f'cos_tilt_{idx}'] = \
                    np.sign(converted_parameters[key])
            else:
                with np.errstate(invalid="raise"):
                    try:
                        converted_parameters[f"cos_tilt_{idx}"] = (
                            converted_parameters[key] / converted_parameters[f"a_{idx}"]
                        )
                    except (FloatingPointError, ZeroDivisionError):
                        logger.debug(
                            f"Error in conversion to spherical spin tilt. "
                            f"This is often due to the spin parameters being zero. "
                            f"Setting cos_tilt_{idx} = 1."
                        )
                        converted_parameters[f"cos_tilt_{idx}"] = 1.0

    # Set default angles
    for key in ["phi_jl", "phi_12"]:
        if key not in converted_parameters:
            converted_parameters[key] = 0.0

    # Convert cosine angles to angles
    for angle in ['tilt_1', 'tilt_2', 'theta_jn']:
        cos_angle = f'cos_{angle}'
        if cos_angle in converted_parameters:
            with np.errstate(invalid="ignore"):
                converted_parameters[angle] = np.arccos(converted_parameters[cos_angle])

    # Handle phase conversion
    if "delta_phase" in original_keys:
        with np.errstate(invalid="ignore"):
            converted_parameters["phase"] = np.mod(
                converted_parameters["delta_phase"]
                - np.sign(np.cos(converted_parameters["theta_jn"]))
                * converted_parameters["psi"],
                2 * np.pi)

    # Generate tidal parameters
    if 'lambda_tilde' in converted_parameters:
        #if converted_parameters.get('system_type', 'BNS').upper() == 'NSBH':
        if converted_parameters.get('lambda_1') == 0.0:
            #print('NSBH case')
            lambda_1, lambda_2 = lambda_tilde_to_lambda_1_lambda_2_NSBH(
                    converted_parameters['lambda_tilde'],
                    converted_parameters['mass_1'],
                    converted_parameters['mass_2']
                )
        else:  # BNS case
            #print('BNS case')
            lambda_1, lambda_2 = lambda_tilde_to_lambda_1_lambda_2(
                    converted_parameters['lambda_tilde'],
                    converted_parameters['mass_1'],
                    converted_parameters['mass_2']
                )
            
        converted_parameters['lambda_1'] = lambda_1
        converted_parameters['lambda_2'] = lambda_2


    # Track which keys were added during conversion
    added_keys = [key for key in converted_parameters.keys()
                 if key not in original_keys]

    return converted_parameters, added_keys

if __name__ == "__main__":
    # Example usage
    priors_class = BNSPriorDict_chirpmass_lambda_tilde(MCL_filename='MCL_BNS_new.dat')
    samples_class = priors_class.sample(100)
    
    plt.figure()
    plt.scatter(samples_class['lambda_tilde'], samples_class['chirp_mass_source'])
    plt.xlabel('Lambda Tilde')
    plt.ylabel('Chirp Mass (source frame)')
    plt.savefig('Mc_relation.png')
    plt.close()