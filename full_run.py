"""
Basically just choose run method. This should be the last step in the chain.

This is only run if we're on the VVO machines. If we're clustered,
run run_driver directly.
"""

import argparse
import subprocess as sp

# Local package files
import grid_search
from run_params import make_diskA_params, make_diskB_params
from constants import today
from tools import already_exists, remove
from sys import version_info; print("Python version: " + str(version_info[:3]))



# If running MCMC, how many processors?
# np = 6

# Which fitting method?
method = 'gs'
m = input("Which type of run?\n['grid', 'mc']: ")
method = 'mc' if 'm' in m else 'gs'

if method == 'gs':
    mol = input('Which spectral line?\n[HCO, HCN, CO, CS]: ').lower()
    if mol in ['hco', 'hcn', 'co', 'cs']:
        diskAParams = make_diskA_params(mol=mol, run_length='long')
        diskBParams = make_diskB_params(mol=mol, run_length='long')

        grid_search.fullRun(diskAParams, diskBParams,
                            mol=mol, cut_central_chans=False)


elif method == 'mc':
    n = input('How many processors shall we use?\n[2-n]: ')
    np = '2' if int(n) < 2 else n

    # sp.call(['mpirun', '-np', np, 'python', 'run_driver.py', '-r'])
    sp.call(['mpirun', '-np', np, 'nice', 'python', 'run_driver.py', '-r'])


elif method == 'fl':
    n = input('How many processors shall we use?\n[2-n]: ')
    np = '2' if int(n) < 2 else n
    print("Using {} processors".format(np))
    sp.call(['mpirun', '-np', np, 'python', 'four_line_run_driver.py', '-r'])


else:
    print('Choose better')

# The End
