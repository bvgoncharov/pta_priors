NANOGrav 15-year data, narrowband, from the standard `par/` and `tim/` folders.

Changes from the original data:

1. Parameters EFAC, EQUAD, ECORR, RNAMP, RNIDX are commented out, so that they are not interpreted as part of the timing model.
2. Some pulsars have `.par` and `.tim` files for all observatories, as well as copies for specific observatories. The latter are deleted, so that there is only one `.par` and `.tim` file per pulsar. 