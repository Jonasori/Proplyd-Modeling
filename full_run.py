"""Basically just choose run method. This should be the last step in the chain.

Trying to get rid of the reliance on mol in constants.
b
"""

import argparse
import subprocess as sp

# Local package files
import grid_search
from run_params import make_diskA_params, make_diskB_params
from constants import today
from tools import already_exists, remove


# If running MCMC, how many processors?
np = 6

# Which fitting method?
method = 'gs'


if method == 'gs':

    mol = raw_input('Which spectral line?\n[HCO, HCN, CO, CS]: ').lower()
    if mol in ['hco', 'hcn', 'co', 'cs']:
        diskAParams = make_diskA_params(mol=mol, run_length='long')
        diskBParams = make_diskB_params(mol=mol, run_length='long')

        grid_search.fullRun(diskAParams, diskBParams,
                            mol=mol, cut_central_chans=False)


elif method == 'fl':
    sp.call(['mpirun', '-np', '4', 'python', 'four_line_run_driver.py', '-r'])



elif method == 'mc':
    sp.call(['mpirun', '-np', '6', 'python', 'run_driver.py', '-r'])



# The End
