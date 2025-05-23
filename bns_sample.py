import bilby
from bilby.core.prior import ConditionalLogUniform, LogUniform, TruncatedGaussian,LogNormal,ConditionalUniform
from bilby.core.prior import PriorDict, Uniform, Constraint,ConditionalInterped,ConditionalPriorDict,Cosine,Sine
import bilby.gw.prior
import numpy as np 
from BNSPriorDict_ChirpMassLambda import BNSPriorDict_chirpmass_lambda_tilde, convert_to_lal_binary_neutron_star_parameters_mchirp

# Setup injection for now 
# Specify the output directory and the name of the simulation.
outdir = "outdir"
# Now we try to sample and see what the error is in generating the probabilies
label = "bns_example"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

# We are going to inject a binary neutron star waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# aligned spins of both black holes (chi_1, chi_2), etc.
injection_parameters = dict(
    mass_1_source=1.5,
    mass_2_source=1.3,
    chi_1=0.02,
    chi_2=0.02,
    luminosity_distance=250.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
    lambda_1=545,
    lambda_2=1346,
)

# We setup the prior dict using the interpolated prior from the file
priors_gw = BNSPriorDict_chirpmass_lambda_tilde(MCL_filename='/Users/smag0001/McLambda_prior/MCL_BNS_new.dat')
# Delta lambda tilde is defined with a uniform prior
priors_gw['delta_lambda_tilde'] = Uniform(name='delta_lambda_tilde', minimum=-5000, maximum=5000, boundary=None)
# Define the other priors for inference 
priors_gw['luminosity_distance'] = 250.0 #bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=1e2, maximum=5e3)
# Fix everything to injected values other than Mchirp and lambda_tilde
priors_gw['dec'] = - 1.2108 #Cosine(name='dec')
priors_gw['ra'] = 1.375 #Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors_gw['theta_jn'] = 0.4 #Sine(name='theta_jn')
priors_gw['psi'] =  2.659 #Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
priors_gw['phase'] = 1.3 #Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors_gw['chi_1'] = 0.02 #bilby.gw.prior.AlignedSpin(name='chi_1', a_prior=Uniform(minimum=0, maximum=0.99))
priors_gw['chi_2'] = 0#bilby.gw.prior.AlignedSpin(name='chi_2', a_prior=Uniform(minimum=0, maximum=0.99))
priors_gw['mass_ratio'] =1.3/1.5 #bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1)
priors_gw['mass_1'] = Constraint(name='mass_1', minimum=0.5, maximum=5)
priors_gw['mass_2'] = Constraint(name='mass_2', minimum=0.5, maximum=5)

# Fix most of the priors to their injected values
for key in [
    "psi",
    "geocent_time",
    "ra",
    "dec",
    "chi_1",
    "chi_2",
    "theta_jn",
    #"luminosity_distance",
    "phase",
]:
    priors_gw[key] = injection_parameters[key]

print(priors_gw)
# Set the duration and sampling frequency of the data segment that we're going
# to inject the signal into. For the
# TaylorF2 waveform, we cut the signal close to the isco frequency
duration = 32
minimum_frequency=500
sampling_frequency = 2048
start_time = injection_parameters["geocent_time"] + 2 - duration

# Fixed arguments passed into the source model. The analysis starts at 40 Hz.
waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2_NRTidal",
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
)

# Create the waveform_generator using a LAL Binary Neutron Star source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    #parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    #parameter_conversion=BNSPriorDict_chirpmass_lambda_tilde.default_conversion_function,
    parameter_conversion=convert_to_lal_binary_neutron_star_parameters_mchirp,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use three interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1), and Virgo (V1)).
# These default to their design sensitivity and start at 40 Hz.
interferometers = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
for interferometer in interferometers:
    interferometer.minimum_frequency = 40
interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration, start_time=start_time
)
interferometers.inject_signal(
    parameters=injection_parameters, waveform_generator=waveform_generator
)

# Initialise the likelihood by passing in the interferometer data (IFOs)
# and the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=interferometers,
    waveform_generator=waveform_generator,
)

# WHY DOES THIS WORK!!!!!!
priors_gw = dict(priors_gw)
#exit()
nsteps = 2000
burnin = nsteps // 3
# Run sampler.  In this case we're going to use the `nestle` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors_gw,
    sampler="nestle",
    nlive=10,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    npool=1,
)

# Convert the posterior using bilby's standard conversion function
result.posterior = bilby.gw.conversion.generate_all_bns_parameters(result.posterior)
#priors = bilby.gw.prior.BNSPriorDict()
# Overwrite custome priors with uniform to save the bilby object 
priors_gw['lambda_tilde'] = Uniform(name='lambda_tilde',minimum=0,maximum=1000)
priors_gw['chirp_mass_source'] = Uniform(name='chirp_mass_source',minimum=1.0,maximum=3.0)

result.plot_corner()
# Create a new serializable result object
serializable_result = bilby.core.result.Result(
    label=label,
    outdir=outdir,
    sampler='nestle',
    search_parameter_keys=['chirp_mass_source', 'mass_ratio', 'lambda_tilde', 'delta_lambda_tilde'],
    fixed_parameter_keys=[],
    priors=priors_gw,
    posterior=result.posterior,
    log_evidence=result.log_evidence,
    log_evidence_err=result.log_evidence_err,
    log_noise_evidence=result.log_noise_evidence,
    log_bayes_factor=result.log_bayes_factor,
    log_likelihood_evaluations=result.log_likelihood_evaluations,
    log_prior_evaluations=result.log_prior_evaluations,
    sampling_time=result.sampling_time,
    meta_data=result.meta_data
)

# Save using standard bilby format
serializable_result.save_to_file()

# Plot the corner plot
#serializable_result.plot_corner()

# Test reading the result back
loaded_result = bilby.core.result.read_in_result(f'{outdir}/{label}_result.json')
print("Successfully loaded result:", loaded_result)