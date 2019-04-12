"""Run the ICR process while cutting off baselines below b_max.

Testing a change.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import argparse as ap
import subprocess as sp
import matplotlib.pyplot as plt
from tools import icr, imstat, already_exists, remove
from constants import today


# baselines = np.arange(0, 130, 5)
baselines = np.arange(0, 155, 5)
default_mol = 'hco'


def get_baseline_rmss(mol, niters=1e4, baselines=baselines, remake_all=False):
    """Iterate through a range of baseline cutoffs and compare the results.

    Args:
        vis (str): the name of the core data file that this is pulling.
        baselines (list of ints): the baselines to check over.
    """
    # Set up the symlink
    run_dir = './data/' + mol + '/baseline_testing/'
    orig_vis = './data/' + mol + '/' + mol
    new_vis = run_dir + mol

    if remake_all is True or already_exists(new_vis) is False:
        sp.call(['mkdir {}'.format(run_dir)], shell=True)

        sp.call(['cp', '-r', '{}.vis'.format(orig_vis),
                 '{}'.format(run_dir)])

    data_list = []
    for b in baselines:
        print('\n\n\n    NEW ITERATION\nBaseline: ', b, '\n')
        name = run_dir + mol + str(b) if b != 0 else run_dir + mol

        # Check if we've already icr'ed this one.
        if already_exists(name + '.cm') is True:
            print("File already exists; going straight to imstat")
            mean, rms = imstat(name, ext='.cm')

        else:
            icr(new_vis, mol=mol, min_baseline=b, niters=niters)
            mean, rms = imstat(name, ext='.cm')

        step_output = {'RMS': rms,
                       'Mean': mean,
                       'Baseline': b}

        data_list.append(step_output)
        print(step_output)

    data_pd = pd.DataFrame(data_list)
    return data_pd


def analysis(df, mol, niters):
    """Read the df from find_baseline_cutoff and do cool shit with it."""
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].grid(axis='x')
    axarr[0].set_title('RMS Noise')
    # axarr[0].set_ylabel('RMS Off-Source Flux (Jy/Beam)')
    # axarr[0].plot(df['Baseline'], df['RMS'], 'or')
    axarr[0].plot(df['Baseline'], df['RMS'], '-b')

    axarr[1].grid(axis='x')
    axarr[1].set_title('Mean Noise')
    axarr[1].set_xlabel('Baseline length (k-lambda)')
    # axarr[1].set_ylabel('Mean Off-Source Flux (Jy/Beam)')
    # axarr[1].plot(df['Baseline'], df['Mean'], 'or')
    axarr[1].plot(df['Baseline'], df['Mean'], '-b')
    im_name = './data/' + mol + '/images/' + mol + '-imnoise.png'
    plt.savefig(im_name)
    # plt.show(block=False)
    return [df['Baseline'], df['Mean'], df['RMS']]


def run_noise_analysis(baselines=baselines, niters=1e4):
    """Run the above functions."""
    print("Baseline range to check: ", baselines[0], baselines[-1])
    print("Don't forget that plots will be saved to /modeling, not here.\n\n")
    mol = input('Which mol?\n').lower()
    ds = get_baseline_rmss(mol, niters, baselines)
    analysis(ds, mol, niters)








def fourmol_analysis(dfs):
    """Read the df from find_baseline_cutoff and do cool shit with it."""
    f, axarr = plt.subplots(2, sharex=True)
    # axarr[0].set_ylabel('RMS Off-Source Flux (Jy/Beam)')
    # axarr[0].plot(df['Baseline'], df['RMS'], 'or')

    colors = sns.diverging_palette(220, 20, n=4, center='dark')
    for df, color in zip(dfs, colors):
        rms_max = max(np.nanmax(df['RMS']), -np.nanmin(df['RMS']))
        mean_max = max(np.nanmax(df['Mean']), -np.nanmin(df['Mean']))
        axarr[0].plot(df['Baseline'], df['RMS']/rms_max,
                      ls='-', lw=2, color=color)
        axarr[1].plot(df['Baseline'], df['Mean']/mean_max,
                      ls='-', lw=2, color=color)

    axarr[0].grid(axis='x')
    axarr[1].grid(axis='x')
    axarr[0].set_title('RMS Noise', weight='bold')
    axarr[1].set_title('Mean Noise', weight='bold')
    axarr[1].set_xlabel('Baseline length (k-lambda)', weight='bold')
    # axarr[1].set_ylabel('Mean Off-Source Flux (Jy/Beam)')
    # axarr[1].plot(df['Baseline'], df['Mean'], 'or')

    image_path = '../Thesis/Figures/full_baseline_analysis.pdf'
    plt.savefig(image_path)
    print('Saved image to ' + image_path)
    # plt.show(block=False)
    # return [df['Baseline'], df['Mean'], df['RMS']]


def run_fourmol_noise_analysis(baselines=baselines, niters=1e4):
    """Run the above functions."""
    print("Baseline range to check: ", baselines[0], baselines[-1])
    print("Don't forget that plots will be saved to /modeling, not here.\n\n")
    dfs = [get_baseline_rmss(mol, niters, baselines)
           for mol in ['hco', 'hcn', 'co', 'cs']]
    fourmol_analysis(dfs)



#"""

def main():
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter,
                               description='''Make a run happen.''')

    parser.add_argument('-r', '--run',
                        action='store_true',
                        help='Run the analysis.')

    parser.add_argument('-o', '--run_and_overwrite',
                        action='store_true',
                        help='Run the analysis, overwriting preexisting runs.')

    args = parser.parse_args()
    if args.run:
        run_noise_analysis(baselines=baselines,
                           niters=1e4)

    elif args.run_and_overwrite:
        run_noise_analysis(default_mol, baselines=baselines,
                           niters=1e4)



if __name__ == '__main__':
    main()

#"""


# The End
