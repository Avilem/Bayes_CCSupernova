import bilby
import numpy as np


def init_sampler(x,prior,model,Dinkpc):
  """
  Function for initializing the priors, and likelihood for the sampler

  x: Time series object from pycbc x = signal + noise
  prior: Choose priors for all parameters
    - Uniform: All except D take values from uniform priors
    - LogUniform: All except D are loguniforms. Tau is still uniform due to negative values
    - triangular: All except D are triangular
    - unif_betasq: Only beta is uniform in beta**2. The rest of the params ~ Uniform 
  """
  priors = bilby.core.prior.PriorDict() #Initialize PriorDict object
  kpc2m = 3.08567758128e+19
  if prior == 'Uniform':
    priors["beta"] = bilby.core.prior.Uniform(0.005, 0.18, "beta",latex_label=r'$\beta$')
    priors["alpha"] = bilby.core.prior.Uniform(0.01, 380, "alpha",latex_label=r'$\alpha$')
    priors["tau"] = bilby.core.prior.Uniform(-6e-4,1e-4, "tau",latex_label=r'$\tau$')
    priors["s"] = bilby.core.prior.Uniform(1e-4, 4e-4, "s",latex_label=r'$s$')
    priors["D"] = Dinkpc * kpc2m
  elif prior == 'LogUniform':
    priors["beta"] = bilby.core.prior.LogUniform(0.005, 0.18, "beta",latex_label=r'$\beta$')
    priors["alpha"] = bilby.core.prior.LogUniform(0.01, 380, "alpha",latex_label=r'$\alpha$')
    priors["tau"] = bilby.core.prior.Uniform(-6e-4,1e-4, "tau",latex_label=r'$\tau$')
    priors["s"] = bilby.core.prior.LogUniform(1e-4, 4e-4, "s",latex_label=r'$s$')
    priors["D"] = Dinkpc * kpc2m
  elif prior == 'triangular':
    priors["beta"] = bilby.core.prior.Triangular((0.18 - 0.005)/2,0.005, 0.18, "beta",latex_label=r'$\beta$')
    priors["alpha"] = bilby.core.prior.Triangular(380/2,0.01, 380, "alpha",latex_label=r'$\alpha$')
    priors["tau"] = bilby.core.prior.Triangular(-2.5e-4,-6e-4,1e-4, "tau",latex_label=r'$\tau$')
    priors["s"] = bilby.core.prior.Triangular(1e-4, 4e-4, "s",latex_label=r'$s$')
    priors["D"] = Dinkpc * kpc2m
  elif prior == 'unif_betasq':
    priors["beta"] = bilby.core.prior.Uniform_betasq(0.005, 0.18, "beta",latex_label=r'$\beta$')
    priors["alpha"] = bilby.core.prior.Uniform(0.01, 380, "alpha",latex_label=r'$\alpha$')
    priors["tau"] = bilby.core.prior.Uniform(-6e-4,1e-4, "tau",latex_label=r'$\tau$')
    priors["s"] = bilby.core.prior.Uniform(1e-4, 4e-4, "s",latex_label=r'$s$')
    priors["D"] = Dinkpc * kpc2m

  sigma_noise = np.std(x) # Using the standard deviation of the whitened and filtered strain
  likelihood = bilby.likelihood.GaussianLikelihood(
        np.array(x.sample_times), 
        np.array(x),
        func=model,
        sigma=sigma_noise,
    )

  return likelihood, priors

