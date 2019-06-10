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
# from sys import version_info; print("Python version: " + str(version_info[:3]))



print('\n\n')

# Which fitting method?
method = 'mc'
# Single or multi-line fit?
single_multi = input("Which type of run?\n['single', 'multi']: ")

if method == 'mc':
    n = input('How many processors shall we use?\n[2-n]: ')
    np = '2' if int(n) < 2 else n

    # sp.call(['mpirun', '-np', np, 'python', 'run_driver.py', '-r'])
    runner = '-rs' if single_multi is 'single' else '-rml'
    sp.call(['mpirun', '-np', np, 'nice', 'python', 'run_driver.py', runner])



elif method == 'gs':
    mol = input('Which spectral line?\n[HCO, HCN, CO, CS]: ').lower()
    if mol in ['hco', 'hcn', 'co', 'cs']:
        diskAParams = make_diskA_params(mol=mol, run_length='long')
        diskBParams = make_diskB_params(mol=mol, run_length='long')

        grid_search.fullRun(diskAParams, diskBParams,
                            mol=mol, cut_central_chans=False)


else:
    print('Choose better')

# The End
