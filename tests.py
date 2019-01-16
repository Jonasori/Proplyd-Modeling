import numpy as np
from analysis import GridSearch_Run
from pathlib2 import Path
import mcmc

Path.cwd()
modeling = '/Volumes/disks/jonas/modeling'
%matplotlib inline


gs_run = GridSearch_Run('gridsearch_runs/jan10_hco/jan10_hco')
len(gs_run.steps)
a, b = gs_run.steps

log = gs_run.depickleLogFile()

gs_run.best_fit_params()



mcmc_run = mcmc.MCMCrun('mcmc_runs/jan10/', 'jan10')

mcmc_run.evolution(save=False)

mcmc_run.evolution_main(save=False)

mcmc_run.corner()





# The End