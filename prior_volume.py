import bilby
from bilby.core.prior import ConditionalLogUniform, LogUniform, TruncatedGaussian,LogNormal,ConditionalUniform
from bilby.core.prior import PriorDict, Uniform, Constraint,ConditionalInterped,ConditionalPriorDict,Cosine,Sine
import bilby.gw.prior
import numpy as np 
import matplotlib.pyplot as plt
from BNSPriorDict_ChirpMassLambda import BNSPriorDict_chirpmass_lambda_tilde, convert_to_lal_binary_neutron_star_parameters_mchirp

# We setup the prior dict using the interpolated prior from the file
priors_physics_based = BNSPriorDict_chirpmass_lambda_tilde(MCL_filename='/Users/smag0001/McLambda_prior/MCL_BNS_new.dat')
print(priors_physics_based)
# # The priors that are not Mchirp/lambda_tilde are set to be the same as the physics based priors
priors_physics_based['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=1e2, maximum=5e3)
priors_physics_based['dec'] = Cosine(name='dec')
priors_physics_based['ra'] = Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors_physics_based['theta_jn'] = Sine(name='theta_jn')
priors_physics_based['psi'] = Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
priors_physics_based['phase'] = Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors_physics_based['chi_1'] = bilby.gw.prior.AlignedSpin(name='chi_1', a_prior=Uniform(minimum=0, maximum=0.99))
priors_physics_based['chi_2'] = bilby.gw.prior.AlignedSpin(name='chi_2', a_prior=Uniform(minimum=0, maximum=0.99))
priors_physics_based['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1)
priors_physics_based['mass_1'] = Constraint(name='mass_1', minimum=0.5, maximum=5)
priors_physics_based['mass_2'] = Constraint(name='mass_2', minimum=0.5, maximum=5)

n_samples = 1000

# Draw samples from the physics based priors 
samples_physics = priors_physics_based.sample(n_samples)

plt.scatter(samples_physics['lambda_tilde'], samples_physics['chirp_mass_source'])
plt.xlabel('Lambda Tilde')
plt.ylabel('Chirp Mass (source frame)')
plt.savefig('Mc_relation.png')


# Setup the uniform priors as well 
priors_uniform = PriorDict()
priors_uniform['lambda_tilde'] = Uniform(name='lambda_tilde', minimum=0.0, maximum=1000.0)
priors_uniform['chirp_mass_source'] = Uniform(name='chirp_mass_source',minimum=0.8,maximum=2.1)
# # The priors that are not Mchirp/lambda_tilde are set to be the same as the physics based priors
priors_uniform['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=1e2, maximum=5e3)
priors_uniform['dec'] = Cosine(name='dec')
priors_uniform['ra'] = Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors_uniform['theta_jn'] = Sine(name='theta_jn')
priors_uniform['psi'] = Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
priors_uniform['phase'] = Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors_uniform['chi_1'] = bilby.gw.prior.AlignedSpin(name='chi_1', a_prior=Uniform(minimum=0, maximum=0.99))
priors_uniform['chi_2'] = bilby.gw.prior.AlignedSpin(name='chi_2', a_prior=Uniform(minimum=0, maximum=0.99))
priors_uniform['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1)
priors_uniform['mass_1'] = Constraint(name='mass_1', minimum=0.5, maximum=5)
priors_uniform['mass_2'] = Constraint(name='mass_2', minimum=0.5, maximum=5)




# Draw samples from the uniform priors 
samples_uniform = priors_uniform.sample(n_samples)

# This is stupid, but we need to do it anyway 
# Drop the samples of RA, DEC, etc. 
keys_to_delete = ['luminosity_distance', 'dec', 'ra', 'theta_jn', 'psi', 'phase', 'chi_1', 'chi_2','mass_ratio']
for key in keys_to_delete:
    del samples_physics[key]
    del samples_uniform[key]
print(samples_physics)
print(samples_uniform)

# We now want to compute the change in prior volume by computing the 90% CI of each samples from the quantiles 

import numpy as np

def calculate_multivariate_ci_volume(samples, ci_level=0.9):
    """
    Calculate the volume of the multivariate confidence interval
    
    Parameters:
    - samples: Array of samples (shape: [num_samples, num_dimensions])
    - ci_level: Confidence interval level (default 0.9 for 90% CI)
    
    Returns:
    - Dictionary with CI volume details
    """
    # Ensure samples is a numpy array
    samples = np.asarray(samples)
    
    # Check sample dimensions
    if samples.ndim != 2:
        raise ValueError("Samples must be a 2D array")
    
    num_samples, num_dimensions = samples.shape
    
    # Compute the kernel density estimate
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(samples.T)
    
    # Compute PDF values for all samples
    pdf_values = kde(samples.T)
    
    # Find the threshold that captures the desired probability mass
    threshold = np.percentile(
        pdf_values, 
        (1 - ci_level) * 100
    )
    
    # Mask samples within the confidence interval
    ci_mask = pdf_values >= threshold
    ci_samples = samples[ci_mask]
    
    # Estimate volume using convex hull
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(ci_samples)
        ci_volume = hull.volume
    except Exception:
        # Fallback to kernel density estimate if convex hull fails
        ci_volume = len(ci_samples) / num_samples
    
    return {
        'volume': ci_volume,
        'threshold': threshold,
        'num_samples_in_ci': np.sum(ci_mask),
        'total_samples': num_samples,
        'ci_samples': ci_samples,
        'bounds': {
            'min': np.min(ci_samples, axis=0),
            'max': np.max(ci_samples, axis=0)
        }
    }
# Function to convert samples dictionary to a 2D NumPy array
def samples_to_2d_array(samples, parameters):
    return np.vstack([samples[param] for param in parameters]).T


# Example usage
def main():
    # Generate some sample multivariate data
    np.random.seed(42)
    

    # Define the parameters to include in the 2D array
    parameters = ['chirp_mass_source', 'lambda_tilde']

    # Convert the samples dictionary to a 2D NumPy array for physics-based samples
    samples_array_physics = samples_to_2d_array(samples_physics, parameters)

    # Convert the samples dictionary to a 2D NumPy array for uniform samples
    samples_array_uniform = samples_to_2d_array(samples_uniform, parameters)
    # Calculate 90% CI
    ci_results = calculate_multivariate_ci_volume(samples_array_physics)

    # Calculate 90% CI uniform 
    ci_results_uniform = calculate_multivariate_ci_volume(samples_array_uniform) 
    
    print("90% Confidence Interval Results:")
    print(f"CI Volume: {ci_results['volume']}")
    delta_prior_volume = ci_results_uniform['volume']-ci_results['volume']
    factor_volume_change = ci_results_uniform['volume']/ci_results['volume']
    print(f"Change in 90% CI volume:{delta_prior_volume}")
    print(f"Threshold: {ci_results['threshold']}")
    print(f"Factor decrease in prior volume:{factor_volume_change}")
    print(f"Samples in CI: {ci_results['num_samples_in_ci']} / {ci_results['total_samples']}")
    print("\nBounds:")
    print("Minimum:", ci_results['bounds']['min'])
    print("Maximum:", ci_results['bounds']['max'])
    
    plt.figure(figsize=(10, 5))
    
    # Scatter plot of all samples
    plt.subplot(121)
    plt.scatter(samples_array_physics[:, 0], samples_array_physics[:, 1], alpha=0.1)
    plt.title('All Samples')
    
    # Scatter plot of CI samples
    plt.subplot(122)
    ci_samples = ci_results['ci_samples']
    plt.scatter(ci_samples[:, 0], ci_samples[:, 1], alpha=0.5)
    plt.title('90% Confidence Interval Samples')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()