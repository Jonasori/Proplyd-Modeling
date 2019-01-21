"""Functions to analyze and plot output from a gridSearch run.

Some thoughts:
    - All of them (so far) have
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess as sp
import matplotlib.pyplot as plt
# import matplotlib.pyplot.RcParams
import matplotlib.gridspec as gridspec

from pathlib2 import Path
from astropy.io import fits
from matplotlib.pylab import *
from matplotlib.ticker import *
from matplotlib.pylab import figure
from matplotlib.patches import Ellipse as ellipse
from astropy.visualization import astropy_mpl_style

from constants import lines, get_data_path, obs_stuff, offsets, get_data_path, mol

sns.set_style('white')
plt.style.use(astropy_mpl_style)
matplotlib.rcParams['font.sans-serif'] = 'Times'
matplotlib.rcParams['font.family'] = 'serif'


Path.cwd()


class GridSearch_Run:
    def __init__(self, path, save_all_plots=False):
        """
        Initialize the object.

        Args:
            path (str): Path to the run dir, plus base name for the files contained therein.
                        Ex: 'gridsearch_runs/jan10_hco/jan10_hco'
            save_all_plots (bool): If true, run all plotting functions and save
                                   the resulting plots.
        """
        self.path = path
        self.mol = self.get_line()

        self.run_date = path.split('/')[-1].split('_')[0]
        self.out_path = './gridsearch_results/' + self.run_date
        self.data_path = './data/{}/{}-short{}.fits'.format(self.mol, self.mol,
                                                            str(lines[mol]['baseline_cutoff']))

        log = self.depickleLogFile()
        self.steps = log[0]
        self.raw_x2 = min(log[1][0])
        self.red_x2 = min(log[1][1])

        self.model_image = fits.getdata(self.path + '_bestFit.fits', ext=0).squeeze()
        self.model_header = fits.getheader(self.path + '_bestFit.fits', ext=0)
        self.data_image = fits.getdata(self.data_path, ext=0).squeeze()
        self.data_header = fits.getheader(self.data_path, ext=0)

        if save_all_plots:
            self.plot_all()



    def get_line(self):
        for mol in ['hco', 'hcn', 'co', 'cs']:
            if mol in self.path:
                break
        return mol


    def depickleLogFile(self):
        """
        Read in the pickle'd full-log file from a run.

        This can be cleaned up significantly, but is functional.
        """
        df = pickle.load(open(self.path + '_step-log.pickle', 'rb'))
        df_a, df_b = df.loc['A', :], df.loc['B', :]
        min_X2_a = min(df_a['Reduced Chi2'])
        min_X2_b = min(df_b['Reduced Chi2'])
        best_fit_a = df_a.loc[df_a['Reduced Chi2'] == min_X2_a]
        best_fit_b = df_b.loc[df_b['Reduced Chi2'] == min_X2_b]
        X2s = [df_a['Raw Chi2'], df_a['Reduced Chi2']]
        del df_a['Reduced Chi2']
        del df_a['Raw Chi2']
        disk_A, disk_B = [], []
        [ disk_A.append({}) for i in df_a ]
        [ disk_B.append({}) for i in df_a ]
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
        return (both_disks, X2s)


    def step_duration(self, ns=[10, 20, 50], save=False):
        """Plot how long each step took, plus some smoothing stuff.

        Args:
            ns (list of ints): A list of the smoothing windows to use.
                               Note len(ns) can't be longer than 5 without adding
                               more colors to colors list.
        """
        plt.close()
        print "\nPlotting step durations..."

        data = pd.read_csv(self.path + '_stepDurations.csv', sep=',')
        xs = data['step']
        ys = data['duration'] / 60

        def get_rolling_avg(xs, ys, n):
            # avg_ys = []
            # for i in range(n / 2, len(ys) - n / 2):
            #    avg_y = sum(ys[i - n / 2:i + n / 2]) / n
            #    avg_ys.append(avg_y)

            avg_ys = [sum(ys[i - n / 2:i + n / 2]) / n
                      for i in range(n / 2, len(ys) - n / 2)]
            return avg_ys


        plt.figure(figsize=(7, 5))
        plt.plot(xs, ys, '-k', linewidth=0.1, label='True time')
        colors = ['orange', 'red', 'blue', 'green', 'yellow']
        for i in range(len(ns)):
            n = ns[i]
            avg_ys = get_rolling_avg(xs, ys, n)
            plt.plot(xs[n / 2:-n / 2], avg_ys, linestyle='-', color=colors[i],
                     linewidth=0.1 * n, label=str(n) + '-step smoothing')

        # Not sure this is right.
        plt.legend()
        plt.xlabel('Step', fontweight='bold')
        plt.ylabel('Time (minutes)', fontweight='bold')
        plt.title('Time per Step for Grid Search Run for ' + self.run_date,
                  fontweight='bold', fontsize=14)
        if save is True:
            plt.savefig(self.out_path + '_durations.pdf')
            print "Saved to " + self.out_path + '_durations.pdf'
        else:
            plt.show()
        plt.clf()


    def best_fit_params(self, save=False):
        """
        Plot where the best-fit values from a grid search fall.

        Plot where the best-fit value(s) stand(s) relative to the range queried in
        a given grid search run.

        Args:
            fname (str): Name of the pickled step log from the grid search.
            Assumes fname is './models/dateofrun/dateofrun'
        """
        plt.close()
        print "\nPlotting best-fit param number lines..."

        run_date = self.run_date
        # both_disks = self.steps

        # Don't plot the parameters that weren't fit.
        # Keep the statics in case we want to do something with them later
        disk_A_full, disk_B_full = self.steps
        disk_A, disk_B = [], []
        disk_A_statics, disk_B_statics = [], []
        for param in disk_A_full:
            if len(param['xvals_queried']) > 1:
                disk_A.append(param)
            else:
                disk_A_statics.append(param)

        for param in disk_B_full:
            if len(param['xvals_queried']) > 1:
                disk_B.append(param)
            else:
                disk_B_statics.append(param)
        both_disks = [disk_A, disk_B]

        colors = ['red', 'blue']
        height = max(len(disk_A), len(disk_B)) + 1

        f, axarr = plt.subplots(height, 2, figsize=[8, height+2])
        axarr[(0, 0)].axis('off')
        axarr[(0, 1)].axis('off')
        axarr[(0, 0)].text(0.2, -0.2, 'Summary of\n' + run_date + ' Run',
                           fontsize=16, fontweight='bold')
        str_rawX2 = str(round(self.raw_x2, 2))
        str_redX2 = str(round(self.red_x2, 6))
        chi_str = '       Min. Raw Chi2: {}\nMin. Reduced Chi2: {}'.format(str_rawX2, str_redX2)
        axarr[(0, 1)].text(0, 0, chi_str, fontsize=10)
        for d in [0, 1]:
            params = both_disks[d]
            for i, p in enumerate(params, 1):
                xs = np.linspace(p['p_min'], p['p_max'], 2)
                axarr[(i, d)].set_title(p['name'], fontsize=10, weight='bold')
                axarr[(i, d)].yaxis.set_ticks([])
                axarr[(i, d)].xaxis.set_ticks(p['xvals_queried'])
                if len(p['xvals_queried']) > 5:
                    axarr[(i, d)].set_xticklabels(p['xvals_queried'],
                                                  rotation=45)
                axarr[(i, d)].plot(xs, [0] * 2, '-k')
                for bf in p['best_fits']:
                    # Make the opacity proportional to how many best fits there are.
                    a = 1 / (2 * len(p['best_fits']))
                    axarr[(i, d)].plot(bf, 0, marker='o', markersize=10,
                                       color='black', alpha=a)
                    axarr[(i, d)].plot(bf, 0, marker='o', markersize=9,
                                       color=colors[d], markerfacecolor='none',
                                       markeredgewidth=3)
                # It'd be nice to not have it fill empty spaces with blank grids.
                # if len(params) < height:


        plt.tight_layout()
        if save is True:
            plt.savefig(self.out_path + '_bestfit_params.pdf')
            print "Saved to " + self.out_path + '_bestfit_params.pdf'
        else:
            plt.show()


    def DMR_images(self, cmap='magma', save=False):
        """Plot a triptych of data, model, and residuals.

        It would be nice to have an option to also plot the grid search results.
        Still need to:
        - Get the beam to plot
        - Get the velocity labels in the right places

        Some nice cmaps: magma, rainbow
        """
        plt.close()
        print "\nPlotting DMR images..."

        model_path = self.path + '_bestFit.fits'
        resid_path = self.path + '_bestFit_resid.fits'
        data_path = self.data_path
        out_path = './gridsearch_results/' + self.run_date + '_DMR-images.pdf'

        real_data = fits.getdata(data_path, ext=0).squeeze()
        image_header = fits.getheader(data_path, ext=0)
        model_data = fits.getdata(model_path, ext=0).squeeze()
        model_header = fits.getheader(model_path, ext=0)
        resid_data = fits.getdata(resid_path, ext=0).squeeze()
        resid_header = fits.getheader(resid_path, ext=0)

        # Define some plotting params
        hspace = -0.2
        wspace = -0.1

        # Set up some physical params
        vmin, vmax = np.nanmin(real_data), np.nanmax(real_data)
        offsets_dA, offsets_dB = offsets[0], offsets[1]
        x_center = int(np.floor(real_data.shape[1] / 2))
        crop_arcsec = 2
        crop_pix = int(crop_arcsec / 0.045)
        xmin, xmax = x_center - crop_pix, x_center + crop_pix
        offsets_dA_pix = [44 - offsets_dA[0]/0.045,
                          44 - offsets_dA[1]/0.045]
        offsets_dB_pix = [44 - offsets_dB[0]/0.045,
                          44 + offsets_dB[1]/0.045]

        chanstep_vel = image_header['CDELT3'] * 0.001
        chan0_vel = image_header['CRVAL3'] * 0.001 - image_header['CRPIX3'] * chanstep_vel
        chan_offset = 15
        nchans = 30

        # Add beam info for the data
        add_beam_d = True if 'bmaj' in image_header else False
        if add_beam_d is True:
            bmin = image_header['bmin'] * 3600.0
            bmaj = image_header['bmaj'] * 3600.0
            bpa = image_header['bpa']

        # Set up which channels are getting plotted, checking to make sure its legal
        if real_data.shape[0] < nchans + chan_offset:
            return 'Aborting; not enough channels to satisfy chan_offset and nchans requested'

        # Add an extra row for the colorbar
        n_rows = int(np.floor(np.sqrt(nchans))) + 1
        n_cols = int(np.ceil(np.sqrt(nchans)))

        # Get the plots going
        # fig = plt.figure(figsize=(n_rows * 3, 7))
        fig = plt.figure(figsize=(18, n_rows + 1))
        big_fig = gridspec.GridSpec(1, 3)

        data_ims = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                                                    subplot_spec=big_fig[0],
                                                    wspace=wspace, hspace=hspace)
        model_ims = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                                                     subplot_spec=big_fig[1],
                                                     wspace=wspace, hspace=hspace)
        resid_ims = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                                                     subplot_spec=big_fig[2],
                                                     wspace=wspace, hspace=hspace)
        # Populate the plots
        print "Got the necessary info; now plotting..."
        for i in range(nchans):
            chan = i + chan_offset
            velocity = str(round(chan0_vel + chan * chanstep_vel, 2))
            ax_d = plt.Subplot(fig, data_ims[i])
            ax_m = plt.Subplot(fig, model_ims[i])
            ax_r = plt.Subplot(fig, resid_ims[i])

            if i == int(np.floor(n_rows / 2)):
                ax_d.set_title('Data', weight='bold')
                ax_m.set_title('Model', weight='bold')
                ax_r.set_title('Residuals', weight='bold')

            # Plot the data
            im_d = ax_d.imshow(real_data[i + chan_offset][xmin:xmax, xmin:xmax],
                               cmap=cmap, vmin=vmin, vmax=vmax)
            im_m = ax_m.imshow(model_data[i + chan_offset][xmin:xmax, xmin:xmax],
                               cmap=cmap, vmin=vmin, vmax=vmax)
            im_r = ax_r.imshow(resid_data[i + chan_offset][xmin:xmax, xmin:xmax],
                               cmap=cmap, vmin=vmin, vmax=vmax)


            # Aesthetic stuff
            # This is all in arcsecs right now. Should be in pix
            # crop_arcsec of 2 translates to 88 pixels across
            # 0, 0 in upper left
            ax_d.grid(False)
            ax_d.set_xticklabels([]), ax_d.set_yticklabels([])
            ax_d.plot(offsets_dA_pix[0], offsets_dA_pix[1], '+g')
            ax_d.plot(offsets_dB_pix[0], offsets_dB_pix[1], '+g')

            ax_m.grid(False)
            ax_m.set_xticklabels([]), ax_m.set_yticklabels([])
            ax_m.plot(offsets_dA_pix[0], offsets_dA_pix[1], '+g')
            ax_m.plot(offsets_dB_pix[0], offsets_dB_pix[1], '+g')

            ax_r.grid(False)
            ax_r.set_xticklabels([]), ax_r.set_yticklabels([])
            ax_r.plot(offsets_dA_pix[0], offsets_dA_pix[1], '+g')
            ax_r.plot(offsets_dB_pix[0], offsets_dB_pix[1], '+g')

            # Add velocity info
            ax_d.text(44, 80, velocity + ' km/s', fontsize=6, color='w',
                    horizontalalignment='center', verticalalignment='center')
            ax_m.text(44, 80, velocity + ' km/s', fontsize=6, color='w',
                    horizontalalignment='center', verticalalignment='center')
            ax_r.text(44, 80, velocity + ' km/s', fontsize=6, color='w',
                    horizontalalignment='center', verticalalignment='center')

            if i == n_rows * (n_cols - 2) and add_beam_d is True:
                el = ellipse(xy=[0.8 * crop_arcsec, 0.8 * crop_pix],
                             width=bmin, height=bmaj, angle=-bpa,
                             fc='k', ec='w', fill=False, hatch='////////')
                ax_d.add_artist(el)

            fig.add_subplot(ax_m)
            fig.add_subplot(ax_d)
            fig.add_subplot(ax_r)
            fig.tight_layout()


        cmaps = imshow(real_data[i + chan_offset][xmin:xmax, xmin:xmax],
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=(crop_arcsec, -crop_arcsec, crop_arcsec, -crop_arcsec))

        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        cax = plt.axes([0.2, 0.06, 0.6, 0.07])
        cbar = colorbar(cmaps, cax=cax, orientation='horizontal')
        cbar.set_label('Jy/beam', labelpad=-12, fontsize=12, weight='bold')
        cbar.set_ticks([vmin, vmax])


        fig.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=0.0, top=0.93)

        # No idea why this is here
        plt.close()

        if save is True:
            fig.savefig(out_path)
            print "Saved to " + out_path
        else:
            print "Showing"
            fig.show()


    def DMR_spectra(self, save=False):
        """
        Plot a model/data/resid triptych of spectra
        y-axis units: each pixel is in Jy/beam, so want to:
            - Multiply each by beam
            - Divide by number of pix (x*y)?
        """
        plt.close()
        print "\nPlotting DMR spectra..."

        model_spec = np.array([np.sum(self.model_image[i])/self.model_image.shape[1]
                               for i in range(self.model_image.shape[0])])
        data_spec = np.array([np.sum(self.data_image[i])/self.data_image.shape[1]
                              for i in range(self.data_image.shape[0])])
        resid_spec = data_spec - model_spec

        chans = np.arange(len(model_spec))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.plot(data_spec, color='steelblue')
        ax2.plot(model_spec, color='steelblue')
        ax3.plot(resid_spec, color='steelblue')

        ax1.set_title('Data', weight='bold')
        ax2.set_title('Model', weight='bold')
        ax3.set_title('Residuals', weight='bold')
        ax1.grid(False), ax2.grid(False), ax3.grid(False)

        ymin = min([min(l) for l in [model_spec, data_spec, resid_spec]])
        ymax = max([max(l) for l in [model_spec, data_spec, resid_spec]])
        ax1.set_xlabel('Channel'), ax1.set_ylabel('Jy/Beam')
        ax1.set_ylim(ymin, ymax), ax2.set_ylim(ymin, ymax), ax3.set_ylim(ymin, ymax)
        plt.tight_layout()
        sns.despine()

        if save:
            plt.savefig(self.out_path + '_DMR-spectra.pdf')
            print "Saved to " + self.out_path + '_DMR-spectra.pdf'
        else:
            plt.show()


    def param_degeneracies(self, DI=0, save=False):
        """
        Plot Chi2 as a function of two params.

        I think this works now.
        """

        df_raw = pickle.load(open(('{}_step-log.pickle').format(self.path), 'rb'))
        df_full = df_raw.loc['A', :] if DI == 0 else df_raw.loc['B', :]
        l = list(df_full.columns)

        for i, p in enumerate(l):
            len_p = len(list(set(df_full[p])))
            if len_p > 1:
                if p != 'Reduced Chi2' and p != 'Raw Chi2':
                    print str([i]), p, '(Length: ' + str(len_p) + ')'

        p1_idx = int(raw_input('Select the index of the first parameter.\n'))
        p2_idx = int(raw_input('Select the index of the second parameter.\n'))
        param1, param2 = l[p1_idx], l[p2_idx]

        # Clear out the parameters that we're not interested in
        l.remove(param1)
        l.remove(param2)
        l.remove('Raw Chi2')
        df = df_full
        for p in l:
            p_vals = list(set(df[p]))
            if len(p_vals) == 1 or p == 'Reduced Chi2':
                df = df.drop(p, axis=1)
            else:
                df = df.drop(df[df[p] != df[p][0]].index)
        df = df.reset_index(drop=True)

        # Make sure we're looking at an iterated parameter.
        len_p1, len_p2 = len(list(set(df[param1]))), len(list(set(df[param2])))
        print len_p1, len_p2
        if len_p1 < 2 or len_p2 < 2:
            return 'Use parameters of length greater than 1'

        """
        I think this was trying to force a landscape layout.
        Could consider using mat.T instead?
        if df[param1][0] == df[param1][1]:
            p_i, p_j = param2, param1
            len_p_i, len_p_j = len_p2, len_p1
        else:
            p_i, p_j = param1, param2
            len_p_i, len_p_j = len_p1, len_p2
        """
        p_i, p_j = param1, param2
        len_p_i, len_p_j = len_p1, len_p2

        # Populate our chi-squared grid
        mat = np.zeros((len_p_i, len_p_j))
        for i in range(len_p_i):
            for j in range(len_p_j):
                this_chi = df['Raw Chi2'][i * len_p_j + j]
                mat[(i, j)] = this_chi
                print this_chi
                print p_i, df[p_i][i * len_p_j + j], '; ', p_j, df[p_j][i * len_p_j + j]
                print

        plt.close()
        # vmin, vmax = np.nanmin(df_full['Raw Chi2']), np.nanmax(df_full['Raw Chi2'])
        plt.matshow(mat, cmap='jet')  #, vmin=vmin, vmax=vmax)
        plt.xlabel(df[p_j].name)
        plt.ylabel(df[p_i].name)
        plt.xticks(range(len_p2), sorted(list(set(df[param2]))))
        plt.yticks(range(len_p1), sorted(list(set(df[param1]))))
        plt.title('Chi2 Map Over Params')
        plt.grid(False, color='k')  #, alpha=0.5)
        plt.colorbar()
        plt.gca().xaxis.tick_bottom()

        if save is True:
            plt.savefig('param_degens.pdf')
        else:
            plt.show(block=False)
        return mat


    def plot_all(self):
        self.best_fit_params(save=True)
        self.step_duration(save=True)
        self.DMR_spectra(save=True)
        self.DMR_images(save=True)









# The End
