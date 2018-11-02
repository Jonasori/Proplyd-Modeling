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

import os
import re
import argparse
import subprocess as sp
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.pylab import *
from matplotlib.ticker import *
from matplotlib.pylab import figure
from matplotlib.patches import Ellipse as ellipse
from astropy.visualization import astropy_mpl_style
from constants import lines, get_data_path, obs_stuff, offsets, get_data_path, mol
plt.style.use(astropy_mpl_style)


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


def plot_fits(image_path, mol=mol, scale_cbar_to_mol=False,
              crop_arcsec=2, cmap='magma',
              save=True, show=True, use_cut_baselines=True):
    """
    Plot some fits image data.

    The cropping currently assumes a square image. That could be easily
    fixed by just adding y_center, y_min, and y_max and putting them in the
    imshow() call.
    Args:
        image_path (str): full path, including filetype, to image.
        crop_arcsec (float): How many arcseconds from 0 should the axis limits be set?
        nchans_to_cut (int): cut n/2 chans off the front and end
        cmap (str): colormap to use. Magma, copper, afmhot, CMRmap, CMRmap(_r) are nice

    Known Bugs:
        - Some values of n_chans_to_cut give a weird error. I don't really wanna
            figure that out right now

    To Do:
        - Maybe do horizontal layout for better screenshots for Evernote.
    """
    image_data = fits.getdata(image_path, ext=0).squeeze()
    header     = fits.getheader(image_path, ext=0)

    if scale_cbar_to_mol is True:
        # Get the data
        dataPath = get_data_path(mol, use_cut_baselines=True)
        real_data = fits.getdata(dataPath + '.fits', ext=0).squeeze()

        vmin = 0
        vmin = np.nanmin(real_data)
        vmax = np.nanmax(real_data)

    else:
        # vmin = -0.25
        vmin = np.nanmin(image_data)
        vmax = np.nanmax(image_data)

    # Add some crosses to show where the disks should be centered (arcsec)
    # offsets_dA, offsets_dB = [-0.0298, 0.072], [-1.0456, -0.1879]
    offsets_dA, offsets_dB = offsets[0], offsets[1]

    # Beam stuff
    add_beam = True if 'bmaj' in header else False
    if add_beam is True:
        bmin = header['bmin'] * 3600.
        bmaj = header['bmaj'] * 3600.
        bpa = header['bpa']

    # Cropping: How many arcsecs radius should it be
    x_center = int(np.floor(image_data.shape[1]/2))
    # If zero is entered, just don't do any cropping and show full image.
    if crop_arcsec == 0:
        crop_arcsec = 256 * 0.045

    crop_pix = int(crop_arcsec / 0.045)
    xmin, xmax = x_center - crop_pix, x_center + crop_pix

    # Set up velocities:
    chanstep_vel = header['CDELT3'] * 1e-3
    # Reference pix value - chanstep * reference pix number
    chan0_vel = (header['CRVAL3'] * 1e-3) - header['CRPIX3'] * chanstep_vel

    # Cut 2/3 of nchans_to_cut out of the front end
    nchans_to_cut=12
    chan_offset = 2 * int(nchans_to_cut/3)
    nchans = image_data.shape[0] - nchans_to_cut

    # I think these are labeled backwards, but whatever.
    n_rows = int(np.floor(np.sqrt(nchans)))
    n_cols = int(np.ceil(np.sqrt(nchans)))
    chan_offset = 4

    fig = figure(figsize=[n_rows, n_cols])

    # Add the actual data
    for i in range(nchans):
        chan = i + int(np.floor(nchans_to_cut/2))
        velocity = str(round(chan0_vel + chan * chanstep_vel, 2))
        # i+1 because plt indexes from 1
        ax = fig.add_subplot(n_cols, n_rows, i+1)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0, -0.8 * crop_arcsec, velocity + ' km/s',
                fontsize=6, color='w',
                horizontalalignment='center', verticalalignment='center')

        if i == n_rows * (n_cols - 1) and add_beam is True:
            el = ellipse(xy=[0.8 * crop_arcsec, 0.8*crop_arcsec],
                         width=bmin, height=bmaj, angle=-bpa,
                         fc='k', ec='w', fill=False, hatch='////////')
            ax.add_artist(el)

        cmaps = imshow(image_data[i + chan_offset][xmin:xmax, xmin:xmax],
                      cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=(crop_arcsec, -crop_arcsec,
                              crop_arcsec, -crop_arcsec))

        ax.plot(offsets_dA[0], offsets_dA[1], '+g')
        ax.plot(offsets_dB[0], offsets_dB[1], '+g')


    # Make the colorbar
    # Want it to be in the gap left at the end of the channels?
    inset_cbar = True
    if inset_cbar is True:
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        cax = plt.axes([0.55, 0.08, 0.38, 0.07])
        cbar = colorbar(cmaps, cax=cax, orientation='horizontal')
        cbar.set_label('Jy/beam',labelpad=-12,fontsize=12, weight='bold')
        cbar.set_ticks([vmin, vmax])

    # If not, put it on top.
    else:
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9)
        cax = plt.axes([0.1, 0.95, 0.81, 0.025])
        cbar = colorbar(cmaps, cax=cax, orientation='horizontal')
        cbar.set_label('Jy/beam',labelpad=-12,fontsize=12, weight='bold')
        cbar.set_ticks([vmin, vmax])


    # Want to save? Want to show?
    if save is True:
        suffix = ''
        if 'data' in image_path:
            resultsPath = 'data/' + mol + '/images/'
            if '-short' in image_path:
                suffix = '-' + image_path.split('-')[1].split('.')[0]
        elif 'mcmc' in image_path:
            resultsPath = 'mcmc_results/'
        elif 'gridsearch' in image_path:
            resultsPath = 'gridsearch_results/'
        else:
            return 'Failed to make image; specify save location.'

        run_name = image_path.split('/')[-2]
        outpath = resultsPath + run_name + suffix + '_image.png'

        plt.savefig(outpath, dpi=200)
        print "Image saved to " + outpath

    if show is True:
        plt.show(block=False)
    plt.gca()


def plot_model_and_data(image_path, mol=mol, scale_cbar_to_mol=False, crop_arcsec=2, cmap='magma', save=True, show=True, use_cut_baselines=True):
    """
    Make a two-panel plot comparing data and model.

    NOT CURRENTLY IMPLEMENTED. NEEDS FULL REWRITE.

    The cropping currently assumes a square image. That could be easily
    fixed by just adding y_center, y_min, and y_max and putting them in the
    imshow() call.
    Args:
        image_path (str): full path, including filetype, to image.
        crop_arcsec (float): How many arcseconds from 0 should the axis limits be set?
        nchans_to_cut (int): cut n/2 chans off the front and end
        cmap (str): colormap to use. Magma, copper, afmhot, CMRmap, CMRmap(_r) are nice

    Known Bugs:
        - Some values of n_chans_to_cut give a weird error. I don't really wanna
            figure that out right now

    To Do:
        - Maybe do horizontal layout for better screenshots for Evernote.
    """
    image_data = fits.getdata(image_path, ext=0).squeeze()
    header     = fits.getheader(image_path, ext=0)

    if scale_cbar_to_mol is True:
        # Get the data
        dataPath = get_data_path(mol, use_cut_baselines=True)
        real_data = fits.getdata(dataPath + '.fits', ext=0).squeeze()

        vmin = np.nanmin(real_data)
        # vmin = -0.5
        vmax = np.nanmax(real_data)

    else:
        vmin = np.nanmin(image_data)
        # vmin = -0.5
        vmax = np.nanmax(image_data)

    # Add some crosses to show where the disks should be centered (arcsec)
    # offsets_dA, offsets_dB = [-0.0298, 0.072], [-1.0456, -0.1879]
    offsets_dA, offsets_dB = offsets[0], offsets[1]

    # Beam stuff
    add_beam = True if 'bmaj' in header else False
    if add_beam is True:
        bmin = header['bmin'] * 3600.
        bmaj = header['bmaj'] * 3600.
        bpa = header['bpa']

    # Cropping: How many arcsecs radius should it be
    x_center = int(np.floor(image_data.shape[1]/2))
    # If zero is entered, just don't do any cropping and show full image.
    if crop_arcsec == 0:
        crop_arcsec = 256 * 0.045

    crop_pix = int(crop_arcsec / 0.045)
    xmin, xmax = x_center - crop_pix, x_center + crop_pix

    # Set up velocities:
    chanstep_vel = header['CDELT3'] * 1e-3
    # Reference pix value - chanstep * reference pix number
    chan0_vel = (header['CRVAL3'] * 1e-3) - header['CRPIX3'] * chanstep_vel

    # Cut 2/3 of nchans_to_cut out of the front end
    nchans_to_cut=12
    chan_offset = 2 * int(nchans_to_cut/3)
    nchans = image_data.shape[0] - nchans_to_cut

    # I think these are labeled backwards, but whatever.
    n_rows = int(np.floor(np.sqrt(nchans)))
    n_cols = int(np.ceil(np.sqrt(nchans)))
    chan_offset = 4

    fig = figure(figsize=[n_rows, n_cols])

    # Add the actual data
    for i in range(nchans):
        chan = i + int(np.floor(nchans_to_cut/2))
        velocity = str(round(chan0_vel + chan * chanstep_vel, 2))
        # i+1 because plt indexes from 1
        ax = fig.add_subplot(n_cols, n_rows, i+1)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0, -0.8 * crop_arcsec, velocity + ' km/s',
                fontsize=6, color='w',
                horizontalalignment='center', verticalalignment='center')

        if i == n_rows * (n_cols - 1) and add_beam is True:
            el = ellipse(xy=[0.8 * crop_arcsec, 0.8*crop_arcsec],
                         width=bmin, height=bmaj, angle=-bpa,
                         fc='k', ec='w', fill=False, hatch='////////')
            ax.add_artist(el)

        cmaps = imshow(image_data[i + chan_offset][xmin:xmax, xmin:xmax],
                      cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=(crop_arcsec, -crop_arcsec,
                              crop_arcsec, -crop_arcsec))

        ax.plot(offsets_dA[0], offsets_dA[1], '+g')
        ax.plot(offsets_dB[0], offsets_dB[1], '+g')


    # Make the colorbar
    # Want it to be in the gap left at the end of the channels?
    inset_cbar = True
    if inset_cbar is True:
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        cax = plt.axes([0.55, 0.08, 0.38, 0.07])
        cbar = colorbar(cmaps, cax=cax, orientation='horizontal')
        cbar.set_label('Jy/beam',labelpad=-12,fontsize=12, weight='bold')
        cbar.set_ticks([vmin, vmax])

    # If not, put it on top.
    else:
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9)
        cax = plt.axes([0.1, 0.95, 0.81, 0.025])
        cbar = colorbar(cmaps, cax=cax, orientation='horizontal')
        cbar.set_label('Jy/beam',labelpad=-12,fontsize=12, weight='bold')
        cbar.set_ticks([vmin, vmax])


    # Want to save? Want to show?
    if save is True:
        suffix = ''
        if 'data' in image_path:
            resultsPath = 'data/' + mol + '/images/'
            if '-short' in image_path:
                suffix = '-' + image_path.split('-')[1].split('.')[0]
        elif 'mcmc' in image_path:
            resultsPath = 'mcmc_results/'
        elif 'gridsearch' in image_path:
            resultsPath = 'gridsearch_results/'
        else:
            return 'Failed to make image; specify save location.'

        run_name = image_path.split('/')[-2]
        outpath = resultsPath + run_name + suffix + '_image.png'

        plt.savefig(outpath, dpi=200)
        print "Image saved to " + outpath

    if show is True:
        plt.show(block=False)
    plt.gca()


def plot_param_degeneracies(dataPath, param1, param2, DI=0):
    df = pickle.load(open('{}_step-log.pickle'.format(dataPath), 'rb'))
    df_a, df_b = df.loc['A', :], df.loc['B', :]
    df = df_a if DI == 0 else df_b

    l = list(df.columns)
    l.remove(param1)
    l.remove(param2)
    l.remove('Reduced Chi2')
    for p in l:
        df = df.drop(df[df[p] != df[p][0]].index)
        df = df.drop(p, axis=1)
    df = df.reset_index(drop=True)

    len_p1, len_p2 = len(list(set(df[param1]))), len(list(set(df[param2])))
    mat = np.zeros((len_p2, len_p1))
    for i in range(len_p1-1):
        for j in range(len_p2-1):
            print i, j, '\t\t', df[param1][i*len_p1], df[param2][j]
            # mat[i, j] = df['Reduced Chi2'][i*len_p1 + j]











# The End
