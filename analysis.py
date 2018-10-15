"""Functions to analyze and plot output from a gridSearch run.

Some thoughts:
    - All of them (so far) have
"""

import pickle
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


matplotlib.rcParams['font.sans-serif'] = "Times"
matplotlib.rcParams['font.family'] = "serif"

resultsPath = '/Volumes/disks/jonas/modeling/gridsearch_results/'

def depickleLogFile(fname):
    """Read in the pickle'd full-log file from a run.

    This can be cleaned up significantly, but is functional.
    """
    df = pickle.load(open('{}_step-log.pickle'.format(fname), 'rb'))
    # Note that we can find the min Chi2 val with:
    # m = df.set_index('Reduced Chi2').loc[min(df['Reduced Chi2'])]
    # This indexes the whole df by RedX2 and then finds the values that
    # minimize that new index.
    # Note, too, that it can be indexed either by slicing or by keys.
    df_a, df_b = df.loc['A', :], df.loc['B', :]
    min_X2_a = min(df_a['Reduced Chi2'])
    min_X2_b = min(df_b['Reduced Chi2'])
    # These come out as length-1 dicts
    best_fit_a = df_a.loc[df_a['Reduced Chi2'] == min_X2_a]
    best_fit_b = df_b.loc[df_b['Reduced Chi2'] == min_X2_b]

    """
    out = {'full_log': df,
           'Disk A log': df_a,
           'Disk B log': df_b,
           'Best Fit A': best_fit_a,
           'Best Fit B': best_fit_b
           }
    """

    # Make one more geared towards plotting
    X2s = [df_a['Raw Chi2'], df_a['Reduced Chi2']]
    del df_a['Reduced Chi2']
    del df_a['Raw Chi2']

    disk_A, disk_B = [], []
    [disk_A.append({}) for i in df_a]
    [disk_B.append({}) for i in df_a]

    for i, p in enumerate(df_a):
        if p != 'Raw Chi2' and p != 'Reduced Chi2':

            ps_A = df_a[p]
            disk_A[i]['p_min'] = min(ps_A)
            disk_A[i]['p_max'] = max(ps_A)
            disk_A[i]['best_fits'] = list(best_fit_a[p])
            disk_A[i]['xvals_queried'] = list(set(ps_A))
            disk_A[i]['name'] = p

            ps_B = df_b[p]
            disk_B[i]['p_min'] = min(ps_B)
            disk_B[i]['p_max'] = max(ps_B)
            disk_B[i]['best_fits'] = list(best_fit_b[p])
            disk_B[i]['xvals_queried'] = list(set(ps_B))
            disk_B[i]['name'] = p

    both_disks = [disk_A, disk_B]
    return both_disks, X2s


def plot_gridSearch_log(fname, show=False):
    """Plot where the best-fit values from a grid search fall.

    Plot where the best-fit value(s) stand(s) relative to the range queried in
    a given grid search run.

    Args:
        fname (str): Name of the pickled step log from the grid search.
        Assumes fname is './models/dateofrun/dateofrun'
    """
    run_date = fname.split('/')[-1]
    # Grab the values to distribute
    both_disks, X2s = depickleLogFile(fname)
    disk_A, disk_B = both_disks
    raw_x2, red_x2 = X2s

    # Plot out
    colors = ['red', 'blue']
    f, axarr = plt.subplots(len(disk_A) + 1, 2, figsize=[8, 8])
    # f, axarr = plt.subplots(len(disk_A), 2, figsize=[5, 8])
    # Add the text info
    axarr[0, 0].axis('off')
    axarr[0, 1].axis('off')
    axarr[0, 0].text(0.2, -0.2, "Summary of\n" + run_date + " Run",
                     fontsize=16, fontweight='bold')

    chi_str = "Min. Raw Chi2: " + str(min(raw_x2)) + \
        "\nMin. Reduced Chi2: " + str(min(red_x2))
    axarr[0, 1].text(0, 0, chi_str, fontsize=10)
    # Plot the number lines
    for d in [0, 1]:
        params = both_disks[d]
        for i, p in enumerate(params, 1):
            # for i, p in enumerate(params):
            xs = np.linspace(p['p_min'], p['p_max'], 2)
            # axarr[i, d].get_xaxis().set_visible(False)
            axarr[i, d].set_title(p['name'], fontsize=10)
            axarr[i, d].yaxis.set_ticks([])
            axarr[i, d].xaxis.set_ticks(p['xvals_queried'])
            axarr[i, d].plot(xs, [0]*2, '-k')
            for bf in p['best_fits']:
                a = 1/(2*len(p['best_fits']))
                axarr[i, d].plot(bf, 0, marker='o', markersize=10,
                                 color='black', alpha=a)
                axarr[i, d].plot(bf, 0, marker='o', markersize=9,
                                 color=colors[d], markerfacecolor='none',
                                 markeredgewidth=3)

    plt.tight_layout()
    plt.savefig(resultsPath + run_date + '_results.png', dpi=200)
    if show is True:
        plt.show()
    plt.clf()


def plot_step_duration(dataPath, ns=[10, 20, 50], show=False):
    """Plot how long each step took, plus some smoothing stuff.

    Args:
        dataPath (str): Path to the run we want to analyze, plus base name,
                        i.e. './models/run_dateofrun/dateofrun'
        ns (list of ints): A list of the smoothing windows to use.
                           Note len(ns) can't be longer than 5 without adding
                           more colors to colors list.
    """
    data = pd.read_csv(dataPath + '_stepDurations.csv', sep=',')
    xs = data['step']
    ys = data['duration']/60

    def get_rolling_avg(xs, ys, n):
        avg_ys = []
        for i in range(n/2, len(ys) - n/2):
            avg_y = sum(ys[i-n/2:i+n/2])/n
            avg_ys.append(avg_y)
        return avg_ys

    plt.figure(figsize=(7,5))
    plt.plot(xs, ys, '-k', linewidth=0.1, label='True time')

    colors = ['orange', 'red', 'blue', 'green', 'yellow']
    for i in range(len(ns)):
        n = ns[i]
        avg_ys = get_rolling_avg(xs, ys, n)
        plt.plot(xs[n/2:-n/2], avg_ys, linestyle='-', color=colors[i],
                 linewidth=0.1 * n, label=str(n) + '-step smoothing')

    run_date = dataPath.split('/')[-1]
    plt.legend()
    plt.xlabel('Step', fontweight='bold')
    plt.ylabel('Time (minutes)', fontweight='bold')
    plt.title('Time per Step for Grid Search Run on ' + run_date,
              fontweight='bold', fontsize=14)

    plt.savefig(resultsPath + run_date + '_durations.png', dpi=200)
    if show is True:
        plt.show()
    plt.clf()


def full_analysis_plot(pickleLog, timeLog):
    """Make a plot with date, chi2 vals, number lines, and time costs.

    Doesn't work right now.
    """
    # Get pickle data
    run_date = pickleLog.split('/')[-1]
    # Grab the values to distribute
    both_disks, X2s = depickleLogFile(pickleLog)
    disk_A, disk_B = both_disks
    raw_x2, red_x2 = X2s
    colors = ['red', 'blue']

    # Get time data
    data = pd.read_csv(timeLog + '_stepDurations.csv', sep=',')
    steps = data['step']
    times = data['duration']/60
    ns = [10, 20, 50]

    def get_rolling_avg(xs, ys, n):
        avg_ys = []
        for i in range(n/2, len(ys) - n/2):
            avg_y = sum(ys[i-n/2:i+n/2])/n
            avg_ys.append(avg_y)
        return avg_ys

    # PLOTTING
    fig = plt.figure(figsize=(7, 15))
    # outer = gridspec.GridSpec(3, 1, wspace=0.2, hspace=0.6)
    outer = gridspec.GridSpec(3, 1, height_ratios=[1, 8, 4], hspace=0.6,
                              wspace=0.2)
    # width_ratios=[1, 1, 1],
    # TOP
    ax_top = plt.Subplot(fig, outer[0])
    ax_top.axis('off')
    ax_top.axis('off')
    ax_top.text(0.2, 0.2, run_date + " Run Summary",
                fontsize=20, fontweight='bold')
    fig.add_subplot(ax_top)

    # outer.tight_layout(fig)
    # MIDDLE
    inner = gridspec.GridSpecFromSubplotSpec(len(both_disks[0]), 2,
                                             subplot_spec=outer[1],
                                             wspace=0.1, hspace=1)
    for d in range(2):
        params = both_disks[d]
        for i, p in enumerate(params):
            xs = np.linspace(p['p_min'], p['p_max'], 2)
            ax = plt.Subplot(fig, inner[i, d])

            ax.set_title(p['name'], fontsize=10)
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticks(p['xvals_queried'])

            ax.plot(xs, [0]*len(xs), '-k')
            for bf in p['best_fits']:
                ax.plot(bf, 0, marker='o', markersize=9,
                        color=colors[d], markerfacecolor='none',
                        markeredgewidth=2)

            fig.add_subplot(ax)
            # fig.sca(inner)
    # BOTTOM
    ax_bottom = plt.Subplot(fig, outer[2])
    ax_bottom.plot(steps, times, '-k', linewidth=0.1, label='True time')

    colors = ['orange', 'red', 'blue', 'green', 'yellow']
    for i in range(len(ns)):
        n = ns[i]
        avg_ys = get_rolling_avg(steps, times, n)
        ax_bottom.plot(steps[n/2:-n/2], avg_ys, linestyle='-', color=colors[i],
                       linewidth=0.1 * n, label=str(n) + '-step smoothing')
    ax_bottom.legend()
    ax_bottom.set_xlabel('Step', fontweight='bold')
    ax_bottom.set_ylabel('Time (minutes)', fontweight='bold')
    ax_bottom.set_title('Time per Step for Grid Search Run on ' + run_date,
                        fontweight='bold', fontsize=14)
    fig.add_subplot(ax_bottom)
    fig.show()


# The End
