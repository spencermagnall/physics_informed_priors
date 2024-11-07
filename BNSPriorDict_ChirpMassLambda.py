import bilby 

from gwpy.timeseries import TimeSeries
from scipy import stats
from scipy.interpolate import interp2d,CloughTocher2DInterpolator

from bilby.core.prior import Prior, Uniform, Interped
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bilby.core.likelihood import GaussianLikelihood
from bilby.core.prior import ConditionalLogUniform, LogUniform,Uniform
from scipy import interpolate
from scipy.signal import find_peaks_cwt
from scipy import signal
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
from bilby.core.prior import ConditionalLogUniform, LogUniform, TruncatedGaussian,LogNormal,ConditionalUniform
from bilby.core.prior import PriorDict, Uniform, Constraint,ConditionalInterped,ConditionalPriorDict,Cosine,Sine
from bilby.gw.prior import UniformInComponentsChirpMass

# Take everything listed above and rap it up into a class, e.g call mchirp_lambda_prior(filename='MCL.dat')
# and it then defines the priors on source frame chirp mass and lambda_tilde and automates the entire process for you
 # Define prior dict 
from bilby.gw.prior import BNSPriorDict
from bilby.core.utils import logger
class BNSPriorDict_chirpmass_lambda_tilde(BNSPriorDict):
    def __init__(self, MCL_filename=None, dictionary=None, filename=None, aligned_spin=True,
                 conversion_function=None):
        """
        Initialize priors for gravitational wave parameters including chirp mass
        and tidal deformability. Priors can be specified either directly through
        parameters or loaded from a file.
        
        Parameters
        ----------


        """

        self.MCL_filename = MCL_filename
        super(BNSPriorDict_chirpmass_lambda_tilde, self).__init__(dictionary=dictionary, filename=filename,
                                           conversion_function=conversion_function)
        

        # Remove unnecessary priors
        for param in ['lambda_1', 'lambda_2', 'chi_1', 'chi_2', 'chirp_mass']:
            self.pop(param, None)
        
        if self.MCL_filename is None:
            print(self.MCL_filename)
            self._setup_default_priors()
        else:
            print(self.MCL_filename)
            self._setup_interpolated_priors(self.MCL_filename)
        

    def _setup_interpolated_priors(self, filename):
        logger.info('Interpolating chirp_mass_source and lambda_tilde prior from file.')
        df = pd.read_csv(filename,delim_whitespace=True)
        #df = df.sort_values(by=[‘mass’])
        mass = df['mass'].tolist()
        tides = df['tides'].tolist()
        prob = df['prob'].tolist()
        mass = np.array(mass)
        tides = np.array(tides)
        prob = np.array(prob)
        # Normalise probability
        prob = prob/np.sum(prob)
        # Create a grid for the points
        points = np.column_stack((mass, tides))
        mass_sort=np.sort(np.unique(mass))
        self.tides_sort=np.sort(np.unique(tides))

        # Set up interpolator
        # We use the CloughTocher2DInterpolator built in to scipy
        # Are there better options?  
        self.f_interp = CloughTocher2DInterpolator(points,prob)

        data = df.to_numpy()
        grid_x_size = mass_sort.shape[0]
        grid_y_size = self.tides_sort.shape[0]
        reshaped_data = data[:,2].reshape((grid_x_size, grid_y_size))

        # The prior on chirp mass is given by marginalising (integrating)
        # the chirp mass-lambda_tilde probabilty across the tidal axis
        chirp_mass_source= np.sum(reshaped_data,axis=0)
        # Normalise
        chirp_mass_source = chirp_mass_source/np.sum(chirp_mass_source)

        # The chirp mass prior is defined as an interpolation prior
        self['chirp_mass_source'] = Interped(xx=mass_sort, yy=chirp_mass_source)

        self['lambda_tilde'] = ConditionalInterped(xx=self.tides_sort, yy=chirp_mass_source,condition_func=self.conditional_func_y)
    
    def __copy__(self):
        """Ensure proper copying of the class"""
        new_obj = type(self)(
            MCL_filename=self.MCL_filename,
            dictionary=self.copy(),
            conversion_function=self.conversion_function
        )
        return new_obj

    def __deepcopy__(self, memodict=None):
        """Ensure proper deep copying of the class"""
        if memodict is None:
            memodict = {}
            
        new_obj = type(self)(
            MCL_filename=self.MCL_filename,
            dictionary=self.copy(),
            conversion_function=self.conversion_function
        )
        memodict[id(self)] = new_obj
        return new_obj

    def _setup_default_priors(self):
        """Set up default uniform priors when no data file is provided."""
        logger.warning('No prior distribution filename given, defaulting to uniform priors.')
        
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

    def conditional_func_y(self,reference_parameters, chirp_mass_source):
        """
        #Condition function for lambda_tilde prior that depends on chirp_mass_source.
        Works with both single values and arrays from the chirp_mass_source Interped prior.
    
        Parameters
        ----------
        reference_parameters : dict
        Dictionary containing original xx and yy values for interpolation
        chirp_mass_source : float or np.ndarray
        Value(s) drawn from the chirp_mass_source Interped prior
        
        Returns
        -------
        dict
        Dictionary with 'yy' containing the interpolated values for lambda_tilde
        """

        tides_sort = self.tides_sort 
        # Handle both single values and arrays from chirp_mass_source prior
        if isinstance(chirp_mass_source, pd.core.series.Series):
            # TODO FIX THIS!!!
            # I want log_prob to be calculated a the end of sampling
            # This is just a hack fix for calculating probability at the end of sampling
            yy = reference_parameters['yy']
        
        else:
            y0_tmp = np.tile(chirp_mass_source, len(tides_sort))

        # TODO Handle negative values, NaN, etc. returned from interpolation
        # Use the interpolation to get lambda_tilde values
        yy = self.f_interp(y0_tmp, tides_sort)

        return dict(yy=yy)
    
if __name__ == "__main__":
    priors_class = BNSPriorDict_chirpmass_lambda_tilde(MCL_filename='MCL_BNS_new.dat')
    samples_class = priors_class.sample(100)
    #print(samples_class)
    plt.scatter(samples_class['lambda_tilde'],samples_class['chirp_mass_source'])
    plt.savefig('Mc_relation.png')