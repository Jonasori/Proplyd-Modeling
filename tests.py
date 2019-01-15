%matplotlib inline

import mcmc
import numpy as np
from analysis import GridSearch_Run
from pathlib2 import Path

Path.cwd()
modeling = '/Volumes/disks/jonas/modeling'

mcmc_run = mcmc.MCMCrun('mcmc_runs/jan10/', 'jan10')

mcmc_run.evolution(save=False)

mcmc_run.evolution_main(save=False)

mcmc_run.corner()


gs_run = GridSearch_Run('gridsearch_runs/jan10_hco/jan10_hco')
str(round(gs_run.red_x2, 2))

gs_run.best_fit_params()







# The End
