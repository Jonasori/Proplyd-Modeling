"""Basically just choose run method. This should be the last step in the chain.

NOTE: To change molecular line, change mol in constants.py.
"""

import argparse
import subprocess as sp

# Local package files
from grid_search import fullRun
from run_params import diskAParams, diskBParams
from constants import today
from tools import already_exists, remove


# If running MCMC, how many processors?
np = 10

#"""
# Which fitting method?
method = 'gs'


if method == 'gs':
    fullRun(diskAParams, diskBParams, cut_central_chans=False)


if method == 'mc':
    sp.call(['mpirun', '-np', '12', 'python', 'run_driver.py', '-r'])
    print "Puppies are nice!\n\n\n"
"""

# ARGPARSE STUFF
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
