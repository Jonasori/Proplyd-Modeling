"""Basically just choose run method. This should be the last step in the chain.

Trying to get rid of the reliance on mol in constants.
"""

import argparse
import subprocess as sp

# Local package files
import grid_search
from run_params import diskAParams, diskBParams
from constants import today
from tools import already_exists, remove


# If running MCMC, how many processors?
np = 6

# Which fitting method?
method = 'gs'


if method == 'gs':
    mol = raw_input('Which spectral line?\n[HCO, HCN, CO, CS]: ').lower()
    if mol in ['hco', 'hcn', 'co', 'cs']:
        grid_search.fullRun(diskAParams, diskBParams,
                            mol=mol, cut_central_chans=False)


elif method == 'fl':
    sp.call(['mpirun', '-np', '4', 'python', 'four_line_run_driver.py', '-r'])



elif method == 'mc':
    sp.call(['mpirun', '-np', '6', 'python', 'run_driver.py', '-r'])




"""
# Argparse stuff, if you want.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the data.')
    parser.add_argument('-gs', '--gridsearch', action='store_true',
                        help='Start a grid search.')

    parser.add_argument('-mc', '--mcmc', action='store_true',
                        help='Start an MCMC run.')

    args = parser.parse_args()
    if args.gridsearch:
        fullRun(diskAParams, diskBParams, cut_central_chans=False)

    if args.mcmc:
        sp.call(['mpirun -np {} python run_driver.py -r'.format(np)])
"""
