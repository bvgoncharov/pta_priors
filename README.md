# pta_priors

Choice of noise priors for PTA GWB searches. 

The code has not been tested as a stand-alone package, please make pull requests with bug fixes you may find (import errors and dependencies are likely issues, but they should be easy to fix).

## Submodules Overview

- **`signal.py`**: Run the standard full-PTA analysis with marginalization over noise hyperparameters, Section 2.2.2 from [arXiv:2409.03661](https://arxiv.org/abs/2409.03661).
- **`noise.py`**: Calculate a marginalized posterior of noise hyperparameters based on single-pulsar analyses, Section 2.2.1 from [arXiv:2409.03661](https://arxiv.org/abs/2409.03661).
- **`quasicommon.py`**: Calculate posteriors on hyperparameters of PTA common red noise based on single-pulsar analyses. Introduced in [arXiv:2206.03766](https://arxiv.org/abs/2206.03766).

## Installation

`python setup.py install --user`

## Example: signal.py

There is a default analysis which is very easy to run, as shown below. It allows to model the distribution (prior) of power-law pulsar spin noise (achromatic red noise) parameters as a truncated Normal distribution without covariance:
```
from pta_priors import signal

# Below, load enterprise PTA object with uniform red noise priors on log10_A and gamma
pta = None

super_model = signal.HierarchicalHyperModel(pta)

# Standard enterprise commands to run the code
sampler = super_model.setup_sampler(resume=True, outdir='output_directory/')
x0 = super_model.initial_sample()
sampler.sample(x0, 1e6, **upd_sample_kwargs)
```
This yields a likelihood marginalised over hyperparameters of the truncated Normal distribution: the mean and the standard deviation. Truncation boundaries are fixed at the positions of uniform prior boundaries. Hyperprior on hyperparameters are chosen to be uniform.

You can model spin noise prior differently, and you can also model priors of other parameters. For example, for DM noise and a uniform (flexible) prior, and a truncated Normal hyperprior, you need to do:
```
from pta_priors import signal
from enterprise.signals import parameter

# Below, load enterprise PTA object with uniform DM noise priors on log10_A and gamma
# (proposal prior)
pta = None

# Let's set the target prior. Dictionary keys should be fragments of enterprise parameter names. Full parameter names will be something like 'J0437-4715_dm_gp_log10_A'.
hierarchical parameters = {
    'dm_gp_log10_A': parameter.UniformPrior,
    'dm_gp_gamma': parameter.UniformPrior
}

# Let's provide samples for hyperparameters of the target prior.
# Keys should correspond to kwargs of the above parameter.UniformPrior;
# Range (0., 10.) is a hyperprior range;
# parameter.TruncatedNormalSampler is a hyperprior distribution.
n_samp = 10000
hyperprior_samples = {
    # Argument order for TruncatedNormalSampler: mean, std, min, max
    'dm_gp_log10_A': {
        'pmin': parameter.TruncatedNormalSampler(-18., 2., -20., -10., size=n_samp)[np.newaxis, :],
        'pmax': parameter.TruncatedNormalSampler(-8., 2., -20., -10., size=n_samp)[np.newaxis, :]
    },
    'dm_gp_gamma': {
        'pmin': parameter.TruncatedNormalSampler(2., 2., 0., 7., size=n_samp)[np.newaxis, :],
        'pmax': parameter.TruncatedNormalSampler(5., 2., 0., 7., size=n_samp)[np.newaxis, :]
    }
} 

super_model = HierarchicalHyperModel(pta, hierarchical_parameters=hierarchical_parameters)

# Set up the sampler and start sampling.
```
The modelling can be performed for any other noise terms. 

## Attribution

If you make use of the code, please cite [arXiv:2409.03661](https://arxiv.org/abs/2409.03661).
<details> 
  <summary>Export BibTeX citation</summary>

> @ARTICLE{GoncharovSardana2024,\
> &nbsp;&nbsp;&nbsp;&nbsp;author = {{Goncharov}, Boris and {Sardana}, Shubhit},\
> &nbsp;&nbsp;&nbsp;&nbsp;title = "{Ensemble noise properties of the European Pulsar Timing Array}",\
> &nbsp;&nbsp;&nbsp;&nbsp;journal = {arXiv e-prints},\
> &nbsp;&nbsp;&nbsp;&nbsp;keywords = {Astrophysics - High Energy Astrophysical Phenomena, Astrophysics - Instrumentation and Methods for Astrophysics},\
> &nbsp;&nbsp;&nbsp;&nbsp;year = 2024,\
> &nbsp;&nbsp;&nbsp;&nbsp;month = sep,\
> &nbsp;&nbsp;&nbsp;&nbsp;eid = {arXiv:2409.03661},\
> &nbsp;&nbsp;&nbsp;&nbsp;pages = {arXiv:2409.03661},\
> &nbsp;&nbsp;&nbsp;&nbsp;doi = {10.48550/arXiv.2409.03661},\
> &nbsp;&nbsp;&nbsp;&nbsp;archivePrefix = {arXiv},\
> &nbsp;&nbsp;&nbsp;&nbsp;eprint = {2409.03661},\
> &nbsp;&nbsp;&nbsp;&nbsp;primaryClass = {astro-ph.HE},\
> &nbsp;&nbsp;&nbsp;&nbsp;adsurl = {[https://ui.adsabs.harvard.edu/abs/2024arXiv240903661G](https://ui.adsabs.harvard.edu/abs/2024arXiv240903661G)},\
> &nbsp;&nbsp;&nbsp;&nbsp;adsnote = {Provided by the SAO/NASA Astrophysics Data System}\
> }
</details>

If you additionally make use of `quasicommon.py`, please cite [arXiv:2206.03766](https://arxiv.org/abs/2206.03766):
<details>
  <summary>Export BibTeX citation</summary>
> @ARTICLE{GoncharovThrane2022,\
> &nbsp;&nbsp;&nbsp;&nbsp;author = {{Goncharov}, Boris and {Thrane}, Eric and {Shannon}, Ryan M. and {Harms}, Jan and {Bhat}, N.~D. Ramesh and {Hobbs}, George and {Kerr}, Matthew and {Manchester}, Richard N. and {Reardon}, Daniel J. and {Russell}, Christopher J. and {Zhu}, Xing-Jiang and {Zic}, Andrew},\
> &nbsp;&nbsp;&nbsp;&nbsp;title = "{Consistency of the Parkes Pulsar Timing Array Signal with a Nanohertz Gravitational-wave Background}",\
> &nbsp;&nbsp;&nbsp;&nbsp;journal = {\apjl},\
> &nbsp;&nbsp;&nbsp;&nbsp;keywords = {Gravitational waves, Millisecond pulsars, Pulsar timing method, Astronomy data analysis, Bayesian statistics, Importance sampling, Supermassive black holes, Gravitational wave astronomy, Hierarchical models, High energy astrophysics, Astronomical methods, 678, 1062, 1305, 1858, 1900, 1892, 1663, 675, 1925, 739, 1043, General Relativity and Quantum Cosmology, Astrophysics - High Energy Astrophysical Phenomena, Astrophysics - Instrumentation and Methods for Astrophysics},\
> &nbsp;&nbsp;&nbsp;&nbsp;year = 2022,\
> &nbsp;&nbsp;&nbsp;&nbsp;month = jun,\
> &nbsp;&nbsp;&nbsp;&nbsp;volume = {932},\
> &nbsp;&nbsp;&nbsp;&nbsp;number = {2},\
> &nbsp;&nbsp;&nbsp;&nbsp;eid = {L22},\
> &nbsp;&nbsp;&nbsp;&nbsp;pages = {L22},\
> &nbsp;&nbsp;&nbsp;&nbsp;doi = {10.3847/2041-8213/ac76bb},\
> &nbsp;&nbsp;&nbsp;&nbsp;archivePrefix = {arXiv},\
> &nbsp;&nbsp;&nbsp;&nbsp;eprint = {2206.03766},\
> &nbsp;&nbsp;&nbsp;&nbsp;primaryClass = {gr-qc},\
> &nbsp;&nbsp;&nbsp;&nbsp;adsurl = {[https://ui.adsabs.harvard.edu/abs/2022ApJ...932L..22G](https://ui.adsabs.harvard.edu/abs/2022ApJ...932L..22G)},\
> &nbsp;&nbsp;&nbsp;&nbsp;adsnote = {Provided by the SAO/NASA Astrophysics Data System}\
> }
</details>
