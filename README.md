Physics Informed priors for neutron star mergers 
=====

Microphysical and astrophysics informed priors for
binary neutron star and neutron star -- black hole mergers

Requirements
------------
We use the following python libaries, all of which are pip installable: 

-  `bilby`
-  `wcosmo`
-  `scipy`
-  `matplotlib`
-  `pandas`
-  `numpy`

Or you can install them all at once using the included requirements.txt 
```bash
pip install -r requirements.txt
```


Usage and Examples
--------

Our priors are designed to be drop in replacements for the standard bilby ``BNSPriorDict``. 
As such, the workflow is almost identical to that of performing parameter estimation with bilby. 
See bns_sample_example.ipynb for a Jupyter notebook that walks you through an example of 
parameter estimation of an injected binary neutron star signal with bilby. 

You should also check out the excellent doccumentation and examples for the bilby code [here](https://lscsoft.docs.ligo.org/bilby/examples.html).

Note that for production runs of BNS mergers you probably want to use the mpi parallel implementation of bilby - [parallel bilby](https://lscsoft.docs.ligo.org/parallel_bilby/).

If you have any issues/comments/queries please contact me at [spencer.magnall@monash.edu](mailto:spencer.magnall@monash.edu)


Citation guide
--------------

Please cite Magnall+ (2025) and [Altiparamak+ (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...939L..34A/abstract) and [Ecker and Rezzola (2022)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.519.2615E/exportcitation) if you use these priors in your work. 
