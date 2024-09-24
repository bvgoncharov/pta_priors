"""
This module extends `enterprise_extensions.HyperModel` to perform 
analytic marginalization over noise hyperparameters. The marginalization 
is based on Section 2.2.2 from Goncharov, Boris, and Shubhit Sardana. 
"Ensemble noise properties of the European Pulsar Timing Array." 
arXiv preprint arXiv:2409.03661 (2024).
"""
import random
import numpy as np
from scipy.special import logsumexp

from enterprise_extensions import hypermodel
from enterprise.signals import parameter


class HierarchicalHyperModel(hypermodel.HyperModel):
    """
    Hierarchical model of pulsar red noise parameters, with 
    analytic-numerical marginalization over noise hyperparameters.

    Parameters
    ----------
    models : dict
        A dictionary of enterprise models, typically keyed by model name.
    log_weights : array-like, optional
        Logarithmic weights for each model, used in model comparison. Default is None.
    hierarchical_parameters : dict, optional
        Dictionary where keys are fragments of parameter names for which hierarchical 
        inference is performed, and values are enterprise.parameter prior functions.
        Default is {'red_noise_log10_A': parameter.TruncNormalPrior, 'red_noise_gamma': parameter.TruncNormalPrior}.
    hyperprior_samples : dict, optional
        Dictionary where keys match those in `hierarchical_parameters` and values are 
        dictionaries containing sampled hyperparameters for the priors. Each entry should 
        be a numpy array of shape (1, N), where N is the number of samples. Default is None.
    n_hyperprior_samples : int, optional
        Number of samples for the hyperpriors if `hyperprior_samples` is not provided. 
        Default is 10,000.

    Raises
    ------
    ValueError
        If the provided `hyperprior_samples` arrays do not have consistent shapes 
        or if they do not follow the expected shape (1, N).
    """
    def __init__(self, models, log_weights=None, hierarchical_parameters=None, hyperprior_samples=None, n_hyperprior_samples=10000):
        if hierarchical_parameters is None:
            hierarchical_parameters = {
                'red_noise_log10_A': parameter.TruncNormalPrior,
                'red_noise_gamma': parameter.TruncNormalPrior
            }

        super().__init__(models, log_weights)

        print('------------------')
        print('Hierarchical Bayesian inference initialization:')

        self.hierarchical_parameters = hierarchical_parameters
        self.hierarchical_parameter_idx = {mm: {} for mm in self.models.keys()}
        self.hierarchical_parameter_mask = {mm: {} for mm in self.models.keys()}
        self.initial_hyperparameters = {mm: {} for mm in self.models.keys()}

        for mm in self.models.keys():
            for p_code in self.hierarchical_parameters.keys():
                param_names = self.models[mm].param_names
                self.hierarchical_parameter_idx[mm][p_code] = [p_code in pp for pp in param_names]
                self.hierarchical_parameter_mask[mm][p_code] = [
                    ii for ii, pp in enumerate(param_names) if p_code in pp
                ]
                self.initial_hyperparameters[mm][p_code] = np.array(
                    self.models[mm].params
                )[self.hierarchical_parameter_mask[mm][p_code]][0].prior.func_kwargs
                print(f'Initial hyperparameters for model {mm}, parameter {p_code}: {self.initial_hyperparameters[mm][p_code]}')

        if hyperprior_samples is None:
            self.hyperprior_sampler = {
                np: samples_truncnorm_based_on_unif(self.models[0].params, noise_parameter=np, n_samples=n_hyperprior_samples)
                for np in hierarchical_parameters.keys()
            }
            self.n_hyperprior_samples = n_hyperprior_samples
        else:
            self.hyperprior_sampler = hyperprior_samples

            # Check for consistency of provided hyperprior samples
            n_samples = [hp_samples.shape for noise_term in hierarchical_parameters.keys() for hp_samples in hyperprior_samples[noise_term].values()]
            unique_shapes = np.unique(n_samples)

            if len(unique_shapes) > 1:
                raise ValueError('All hyperprior_samples must have consistent number of samples across noise terms. Array shapes provided: ', unique_shapes)
            elif unique_shapes[0][0] != 1:
                raise ValueError('Each hyperprior_sample array must have shape (1, N).')
            else:
                self.n_hyperprior_samples = unique_shapes[0][1]

        print('------------------')

    def hyperprior_log_weight_factor(self, x, nmodel):
        """
        Calculate the log weight factor due to the hyperprior in the likelihood.

        Parameters
        ----------
        x : array-like
            Current parameter values.
        nmodel : int
            Index of the active model.

        Returns
        -------
        float
            The log weight factor from the hyperprior.
        """
        log_prior_ratio_total = 0.0

        for p_code, p_prior in self.hierarchical_parameters.items():
            pmask = self.hierarchical_parameter_mask[nmodel][p_code]
            theta = np.array(x)[pmask][:, np.newaxis]  # Convert to column vector
            log_priors_target = np.log(p_prior(theta, **self.hyperprior_sampler[p_code]))
            log_priors_proposal = np.array([psr_param.get_logpdf(pval) for psr_param, pval in zip(np.array(self.models[nmodel].params)[pmask], np.array(x)[pmask])])[:, np.newaxis]

            # Sum over pulsars: \pi(\theta) = \prod \theta_i
            log_prior_ratio_total += np.sum(log_priors_target - log_priors_proposal, axis=0)

        return logsumexp(log_prior_ratio_total) - np.log(self.n_hyperprior_samples)

    def get_lnlikelihood(self, x):
        """
        Compute the log likelihood for the current set of parameters, 
        including the hyperprior log weight factor.

        Parameters
        ----------
        x : array-like
            Current parameter values.

        Returns
        -------
        float
            The log likelihood including the model and hyperprior contributions.
        """
        # Find model index variable
        idx = self.param_names.index('nmodel')
        nmodel = int(np.rint(x[idx]))

        # Find parameters of the active model
        q = [x[self.param_names.index(par)] for par in self.models[nmodel].param_names]

        # Compute log likelihood for the active model, including the hyperprior factor
        active_lnlike = self.models[nmodel].get_lnlikelihood(q) + self.hyperprior_log_weight_factor(q, nmodel)

        # Add log weights if they exist
        if self.log_weights is not None:
            active_lnlike += self.log_weights[nmodel]

        return active_lnlike

def samples_truncnorm_based_on_unif(params, sigma_range=[0., 10.], n_samples=10000, noise_parameter='red_noise_log10_A'):
    """
    Generates samples from a truncated normal distribution for a target parameter 
    using a uniform proposal distribution's hyperparameters.

    Parameters
    ----------
    params : list
        A list of parameter objects from an enterprise model.
    sigma_range : list, optional
        Range for sigma (standard deviation) sampling, default is [0., 10.].
    n_samples : int, optional
        Number of samples to generate, default is 10,000.
    noise_parameter : str, optional
        Fragment of the name of the noise parameter to filter the relevant parameter 
        from the input list. Default is 'red_noise_log10_A'.

    Returns
    -------
    dict
        A dictionary with keys 'mu', 'sigma', 'pmin', and 'pmax', each containing 
        arrays of shape (1, n_samples) with sampled hyperparameter values:
            - 'mu': Mean of the truncated normal, sampled uniformly between 
              pmin and pmax of the uniform proposal distribution.
            - 'sigma': Standard deviation, sampled uniformly from the provided sigma_range.
            - 'pmin': Minimum bound of the uniform prior.
            - 'pmax': Maximum bound of the uniform prior.
    """
    # Select a random parameter object matching the noise_parameter
    param = random.choice([pp for pp in params if noise_parameter in pp.name])

    # Extract pmin and pmax from the proposal's uniform prior bounds
    pmin = param.prior._defaults['pmin']
    pmax = param.prior._defaults['pmax']

    # Generate samples for mean (mu) from a uniform distribution between pmin and pmax
    mu_samples = parameter.UniformSampler(pmin, pmax, size=n_samples)[np.newaxis, :]

    # Generate samples for standard deviation (sigma) uniformly from the given sigma_range
    sigma_samples = parameter.UniformSampler(sigma_range[0], sigma_range[1], size=n_samples)[np.newaxis, :]

    # Return a dictionary with hyperparameter samples
    return {
        'mu': mu_samples,
        'sigma': sigma_samples,
        'pmin': np.repeat(pmin, n_samples)[np.newaxis, :],
        'pmax': np.repeat(pmax, n_samples)[np.newaxis, :]
    }
