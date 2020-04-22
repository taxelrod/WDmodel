#! /bin/bash
set -ev

export PYSYN_CDBS=/home/tsa/Dropbox/WD/PyWD/WDmodel/WDdata/photometry/synphot/

./fit_WDmodel --specfile tests/test.flm --photfile tests/test.phot --rebin 2 --redo --trimspec 3700 5200 --samptype pt --ntemps 2 --nburnin 20 --nprod 2000 --nwalkers 100 --covtype SHO --tau_fix True --tau 5000 --lamshift 2 --vel -50 --reddeningmodel f99 --excludepb F160W --rescale --blotch --phot_dispersion 0.001 --thin 2 --dl 380 --trimspec 3700 5200 --av_spec 0.21
