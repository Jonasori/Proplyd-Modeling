"""
Functions to analyze and plot output from a gridSearch run.

IDEA: Add molecular line name to the data (and model) headers. Would save a lot of
        string parsing hassle. Would've been nice to think of this a couple
        months ago.

To-Do: In GridSearch_Run.param_degeneracies(), choose which slice of the
       non-plotted params we're looking through. Right now it's not doing that,
       so it's not showing the best-fit point in p-space.
"""

import os, yaml, emcee, pickle, argparse, matplotlib
import numpy as np, numpy.ma as ma, pandas as pd, seaborn as sns
import subprocess as sp, astropy.units as u

from disk_model3.disk import Disk
from astropy.utils import data
from astropy.io import fits
from pathlib2 import Path
from yaml import CLoader, CDumper

# Yuck
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from matplotlib.pylab import *
from matplotlib.pylab import figure
from matplotlib.ticker import *
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator, LinearLocator, AutoMinorLocator
from matplotlib.patches import Ellipse
from astropy.visualization import astropy_mpl_style
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Local Package Imports
import utils
import run_driver

# Consolidate these two
import tools
from tools import imstat, imstat_single, pipe, moment_maps
# from constants import lines, get_data_path, obs_stuff, offsets, get_data_path, mol


# https://media.readthedocs.org/pdf/spectral-cube/latest/spectral-cube.pdf
# from spectral_cube import SpectralCube

import sys
sys.version

os.chdir('/Volumes/disks/jonas/modeling')
Path.cwd()

sns.set_style('white')
plt.style.use(astropy_mpl_style)
matplotlib.rcParams['font.sans-serif'] = 'Times'
matplotlib.rcParams['font.family'] = 'serif'





class MCMC_Analysis:
    """
    A data object for an MCMC run with some plotting and analysis tools.

    This takes an existing run (either still running or completed) and allows
    us to make some pretty figures from it.
    """
    # Set up a path for the images to go to:
    def __init__(self, run_name, burn_in=0, print_stats=False):
        """Set up.

        Args:
            run_name (str): the name of the run (for output files;
                            usually just '[today]-[run number of the day]').
            burn_in (int): how many of the first steps to ignore.
        """


        self.name            = run_name
        self.runpath         = './mcmc_runs/{}/'.format(self.name)
        self.mol             = self.get_line()
        self.main            = pd.read_csv(self.runpath + self.name + '_chain.csv')

        if self.mol is 'multi':
            self.param_dicts = {'hco': yaml.load(open('{}params-hco.yaml'.format(self.runpath, self.mol),
                                                  'r'), Loader=CLoader),
                                'hcn': yaml.load(open('{}params-hcn.yaml'.format(self.runpath, self.mol),
                                                                      'r'), Loader=CLoader)
                                }

            # Build a combined param_dict.
            self.param_dict = {}
            for k in list(self.param_dicts['hco'].keys()):
                # If an element is held constant, it has the same value in both
                # so just add it straight. Otherwise, label it.
                if self.param_dicts['hco'][k] == self.param_dicts['hcn'][k]:
                    self.param_dict[k] = self.param_dicts['hco'][k]
                else:
                    self.param_dict[k + '_hco'] = self.param_dicts['hco'][k]
                    self.param_dict[k + '_hcn'] = self.param_dicts['hcn'][k]

            # print('New param dict keys:')
            # print([k for k in list(self.param_dict.keys())])
            self.nwalkers    = self.param_dict['nwalkers']
            self.data_paths  = {'hco': './data/hco/hco-short110',
                                'hcn': './data/hcn/hcn-short80'}


        else:
            self.param_dict  = yaml.load(open('{}params-{}.yaml'.format(self.runpath, self.mol),
                                                  'r'), Loader=CLoader)
            self.nwalkers = self.param_dict['nwalkers']
            self.uv_cut = self.param_dict['baseline_cutoff']
            self.data_path  = './data/{}/{}-short{}'.format(self.mol, self.mol,
                                                            self.uv_cut)

        self.nsteps = self.main.shape[0] // self.nwalkers
        self.burnt_in = self.main.iloc[burn_in*self.nwalkers:, :]

        # Get rid of steps that resulted in bad lnprobs
        lnprob_vals = self.burnt_in.loc[:, 'lnprob']
        self.groomed = self.burnt_in.loc[lnprob_vals != -np.inf,
                                         :].reset_index().drop('index', axis=1)
        # print 'Removed burn-in phase (step 0 through {}).'.format(burn_in)

        # This isn't set up for multi-line yet
        self.plot_labels_dict = {'atms_temp_A': 'Atms. Temp, 150 au\n(Disk A)',
                                 'atms_temp_B': 'Atms. Temp, 150 au\n(Disk B)',
                                 'mol_abundance_A': 'Log Abundance\n(Disk A)',
                                 'mol_abundance_B': 'Log Abundance\n(Disk B)',
                                 'pos_angle_A': 'Position Angle\n(Disk A)',
                                 'pos_angle_B': 'Position Angle\n(Disk B)',
                                 'r_out_A': 'Outer Radius\n(Disk A)',
                                 'r_out_B': 'Outer Radius\n(Disk B)',
                                 'incl_A': 'Inclination\n(Disk A)',
                                 'incl_B': 'Inclination\n(Disk B)',
                                 'm_disk_A': 'Log Mass\n(Disk A)',
                                 'm_disk_B': 'Log Mass\n(Disk B)',
                                 'temp_struct_A': 'q\n(Disk A)',
                                 'temp_struct_B': 'q\n(Disk B)'}


        print("\n\n" + self.name + "currently has {} steps over {} walkers.\n\n".format(str(self.nsteps), str(self.nwalkers)))

        self.get_bestfit_dict()  # yields self.bf_param_dict
        self.get_fit_stats(print_stats=print_stats)




    def get_line(self):
        for mol in ['hco', 'hcn', 'co', 'cs', 'multi']:
            if mol in self.runpath:
                break
        return mol


    def get_bestfit_dict(self):
        subset_df = self.main
        # Locate the best fit model from max'ed lnprob.
        max_lnp = subset_df['lnprob'].max()
        model_params = subset_df[subset_df['lnprob'] == max_lnp].drop_duplicates()

        self.bf_param_dict = self.param_dict.copy()
        for param in model_params.columns[:-1]:
            self.bf_param_dict[param] = model_params[param].values[0]
        yaml.dump(self.bf_param_dict,
                  open('{}params-{}_bestFit.yaml'.format(self.runpath, self.mol), 'w+'),
                  Dumper=CDumper)
        print("Dumped best-fit param dict to {}params-{}_bestFit.yaml".format(self.runpath, self.mol))
        return None


    def get_fit_stats(self, print_stats):
        stats = self.groomed.describe(percentiles=[0.16, 0.84]).drop([
            'count', 'min', 'max', 'mean'])
        stats.loc['best fit'] = self.main.loc[self.main['lnprob'].idxmax()]
        stats = stats.iloc[[-1]].append(stats.iloc[:-1])
        stats.loc[['16%', '84%'], :] -= stats.loc['50%', :]
        stats = stats.reindex(
            ['50%', '16%', '84%', 'best fit', 'std'], copy=False)
        self.fit_stats = stats.T.round(6)

        if print_stats:
            print(stats.T.round(6).to_string())


    # THIS IS VERY INCOMPLETE
    def get_disk_objects(self, set_face_on=False):
        self.get_bestfit_dict()
        param_dict = self.bf_param_dict
        if set_face_on:
            param_dict['incl_A'] = 0
            param_dict['incl_B'] = 0

        mols = ['hco', 'hcn'] if self.mol is 'multi' else self.mol
        self.diskA, self.diskB = [], []
        for m in mols:


            # Generate a Disk object
            DI = 0
            d1 = Disk(params=[param_dict['temp_struct_A'],
                              10**param_dict['m_disk_A'],
                              param_dict['surf_dens_str_A'],
                              param_dict['r_ins'][DI],
                              param_dict['r_out_A'],
                              param_dict['r_crit'],
                              param_dict['incl_A'],
                              param_dict['m_stars'][DI],
                              10**param_dict['mol_abundance_A'],
                              param_dict['v_turb'],
                              param_dict['vert_temp_str'],
                              param_dict['T_mids'][DI],
                              param_dict['atms_temp_A'],
                              param_dict['column_densities'],
                              [param_dict['r_ins'][DI], param_dict['r_out_A']],
                              param_dict['rot_hands'][DI]],
                      rtg=False)
            d1.Tco = param_dict['T_freezeout']
            d1.set_rt_grid()
            # Now do Disk 2
            DI = 1
            d2 = Disk(params=[param_dict['temp_struct_B'],
                              10**param_dict['m_disk_B'],
                              param_dict['surf_dens_str_B'],
                              param_dict['r_ins'][DI],
                              param_dict['r_out_B'],
                              param_dict['r_crit'],
                              param_dict['incl_B'],
                              param_dict['m_stars'][DI],
                              10**param_dict['mol_abundance_B'],
                              param_dict['v_turb'],
                              param_dict['vert_temp_str'],
                              param_dict['T_mids'][DI],
                              param_dict['atms_temp_B'],
                              param_dict['column_densities'],
                              [param_dict['r_ins'][DI], param_dict['r_out_B']],
                              param_dict['rot_hands'][DI]],
                      rtg=False)
            d2.Tco = param_dict['T_freezeout']
            d2.set_rt_grid()

            self.diskA.append(d1)
            self.diskB.append(d2)

        return None


    def evolution(self, save=True):
        """ Plot walker evolution. """
        print('Making walker evolution plot...')
        plt.close()

        self.nsteps = len(self.groomed)//self.nwalkers
        stepmin, stepmax = 0, self.nsteps
        print(self.nsteps, self.nwalkers)

        main = self.groomed.copy().iloc[stepmin * self.nwalkers:
                                        stepmax * self.nwalkers, :]

        # Horrifying, but maybe functional.
        base = main.iloc[0::self.nwalkers].assign(step=np.arange(stepmin, stepmax))
        axes = base.plot(x='step', figsize=(9, 1.5*(len(main.columns))),
                         subplots=True, color='black', alpha=0.1, legend=False)

        # Add in plot titles
        for i in range(len(axes)):
            bf_val = main[main['lnprob'] == np.nanmax(main['lnprob'])][main.columns[i]].values[0]
            title_str = "{}: {}".format(main.columns[i], round(bf_val, 2))
            axes[i].set_title(title_str, # weight='bold',
                              fontdict={'fontsize': 12}, loc='right')

        # Populate the walker plots
        for i in range(self.nwalkers-1):
            step_walkers = main.iloc[i+1::self.nwalkers].assign(Step=np.arange(stepmin, stepmax))
            step_walkers.plot(x='Step', subplots=True, ax=axes,
                      legend=False, color='black', alpha=0.1)

            amin = main.iloc[-1 * self.nwalkers:, -1].min()
            amax = main.lnprob.max()
            axes[-1].set_ylim(amin, amax)
            #axes[-1].set_ylim(-40000, 0)

        # if you want mean at each step over plotted:
        main.index //= self.nwalkers
        walker_means = pd.DataFrame([main.loc[i].mean() for i in range(self.nsteps)])
        walker_means.plot(subplots=True, ax=axes, legend=False,
                          color='forestgreen', ls='--', linewidth=0.75)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.suptitle(self.name + ' walker evolution', weight='bold')
        if save:
            plt.savefig(self.image_outpath + '_evolution.pdf')
            print('Image saved image to ' + self.image_outpath + '_evolution.pdf')
        else:
            plt.show()


    def kde(self):
        """Make a kernel density estimate (KDE) plot."""
        print('Generating posterior kde plots...')
        plt.close()

        nrows, ncols = (2, int(np.ceil((self.groomed.shape[1] - 1) / 2.)))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5*ncols, 2.5*nrows))

        # plot kde of each free parameter
        for i, param in enumerate(self.groomed.columns[:-1]):
            ax = axes.flatten()[i]

            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
            ax.set_title(param)
            ax.tick_params(axis='y', left='off', labelleft='off')

            samples = self.groomed.loc[:, param]
            plotting.my_kde(samples, ax=ax)

            percentiles = samples.quantile([.16, .5, .84])
            ax.axvline(percentiles.iloc[0], lw=1,
                       ls='dotted', color='k', alpha=0.5)
            ax.axvline(percentiles.iloc[1], lw=1.5, ls='dotted', color='k')
            ax.axvline(percentiles.iloc[2], lw=1,
                       ls='dotted', color='k', alpha=0.5)

        # hide unfilled axes
        for ax in axes.flatten()[self.groomed.shape[1]:]:
            ax.set_axis_off()

        # bivariate kde to fill last subplot
        # ax = axes.flatten()[-1]
        # for tick in ax.get_xticklabels(): tick.set_rotation(30)
        # sns.kdeplot(self.groomed[r'$i$ ($\degree$)'], self.groomed[r'Scale Factor'], shade=True, cmap='Blues', n_levels=6, ax=ax);
        # ax.tick_params(axis='y', left='off', labelleft='off', right='on', labelright='on')

        # adjust spacing and save
        plt.tight_layout()
        plt.savefig(self.image_outpath + '_kde.png', dpi=200)
        print('Image saved image to ' + self.image_outpath + '_kde.png')
        plt.show()

    # This needs multi-line updating
    def posteriors(self, save=False, save_to_thesis=False):
        """Plot just the walkers'  posteriors (i.e. the diagonal of a corner plot)."""
        df = self.groomed #.drop('lnprob', axis=1)

        df_a_params = [p for p in df.columns if '_A' in p]
        df_b_params = [p for p in df.columns if '_B' in p]

        n_cols = max((len(df_a_params), len(df_b_params)))

        plt.close()
        fig, axes = plt.subplots(2, n_cols,
                                 figsize=(3*n_cols, 5))

        for i in range(2):
            for ax, param in zip(axes[i], df_a_params):
                print("Adding posterior for {} to plot.".format(param))
                sns.distplot(df[param], kde=True, ax=ax)
                xlab = ax.xaxis.get_label().get_text()
                ax.set_xlabel(self.plot_labels_dict[xlab]) #, weight='bold')

                # Plot best-fit, 50th and +/- 1 sigma lines
                x = self.bf_param_dict[param]
                x_bf = self.fit_stats['best fit'][param]
                x_fiftieth = self.fit_stats['50%'][param]
                x_minus = self.fit_stats['50%'][param] + self.fit_stats['16%'][param] # this is negative, so don't worry
                x_plus = self.fit_stats['50%'][param] + self.fit_stats['84%'][param]
                ymin, ymax = ax.get_ylim()
                ax.plot((x_bf, x_bf), (ymin, ymax), '--r', lw=4)
                ax.plot((x_fiftieth, x_fiftieth), (ymin, ymax), linestyle=':', color='dimgray', lw=3)
                ax.plot((x_minus, x_minus), (ymin, ymax), linestyle=':', color='gray', lw=2)
                ax.plot((x_plus, x_plus), (ymin, ymax), linestyle=':', color='gray', lw=2)


        # If there are empties, delete them.
        # This hardcodes in the assumption that disk B's list is shorter.
        empties = len(axes[1]) - len(df_b_params)
        for i in range(len(axes[1]) - empties, len(axes[1])):
            fig.delaxes(axes[1][i])

        if self.mol is 'hco':
            mol_name = r"HCO$^+$(4-3)"
        else:
            j = lines[self.mol]['jnum']
            mol_name = self.mol.upper() + "({}-{})".format(j, j-1)

        plt.suptitle('{} MCMC Posteriors'.format(mol_name), weight='bold')
        plt.tight_layout()

        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9,
                            wspace=None, hspace=None)

        if save:
            if save_to_thesis:
                image_outpath = '../Thesis/Figures/posteriors_{}.pdf'.format(self.mol)
            else:
                image_outpath = './mcmc_runs/{}-posteriors.pdf'.format(self.name)
            plt.savefig(image_outpath)
            print("Saved image to " + image_outpath)
        else:
            plt.show(block=False)


    def corner(self, variables=None, save=True, save_to_thesis=False):
        """Plot corner plot of fit."""
        plt.close()

        # get best_fit and posterior statistics
        stats = self.groomed.describe(percentiles=[0.16, 0.84]).drop([
            'count', 'min', 'max', 'mean'])
        stats.loc['best fit'] = self.main.loc[self.main['lnprob'].idxmax()]
        stats = stats.iloc[[-1]].append(stats.iloc[:-1])
        stats.loc[['16%', '84%'], :] -= stats.loc['50%', :]
        stats = stats.reindex(
            ['50%', '16%', '84%', 'best fit', 'std'], copy=False)
        print((stats.T.round(6).to_string()))

        # make corner plot
        print("Starting SNS PairGrid")
        # corner_plt = sns.PairGrid(data=self.groomed, diag_sharey=False, despine=False, vars=variables)
        corner_df = self.groomed.drop('lnprob', axis=1)
        corner_plt = sns.PairGrid(data=corner_df, diag_sharey=False, despine=False, vars=variables)

        print("Finished SNS PairGrid")

        if variables is not None:
            print("Entering if")
            corner_plt.map_lower(plt.scatter, s=1, color='#708090', alpha=0.1)
        else:
            print("Entering else. This is the part that takes forever.")
            # corner_plt.map_lower(sns.kdeplot, cut=0, cmap='Blues',
            #                      n_levels=18, shade=True)
            corner_plt.map_lower(sns.kdeplot, cut=0, cmap='Blues',
                                 n_levels=5)

        # This is where the error is coming from:
        # ValueError: zero-size array to reduction operation minimum which has no identity
        # corner.map_lower(sns.kdeplot, cut=0, cmap='Blues', n_levels=3, shade=False)
        corner_plt.map_diag(sns.kdeplot, cut=0)

        if save and save_to_thesis:
            # get best_fit and posterior statistics
            stats = self.groomed.describe(percentiles=[0.16, 0.84]).drop([
                'count', 'min', 'max', 'mean'])
            stats.loc['best fit'] = self.main.loc[self.main['lnprob'].idxmax()]
            stats = stats.iloc[[-1]].append(stats.iloc[:-1])
            stats.loc[['16%', '84%'], :] -= stats.loc['50%', :]
            stats = stats.reindex(
                ['50%', '16%', '84%', 'best fit', 'std'], copy=False)
            print((stats.T.round(3).to_string()))
            # print(stats.round(2).to_latex())

            # add stats to corner plot as table
            # table_ax = corner_plt.fig.add_axes([0, 0, 1, 1], frameon=False)
            # table_ax.axis('off')
            # left, bottom = 0.15, 0.83
            # pd.plotting.table(table_ax, stats.round(2), bbox=[
            #                   left, bottom, 1-left, .12], edges='open', colLoc='right')
            #
            # corner_plt.fig.suptitle(r'{} Parameters, {} Walkers, {} Steps $\to$ {} Samples'
            #                     .format(self.groomed.shape[1], self.nwalkers,
            #                             self.groomed.shape[0]//self.nwalkers, self.groomed.shape[0],
            #                             fontsize=25))


        # hide upper triangle, so that it's a conventional corner plot
        for i, j in zip(*np.triu_indices_from(corner_plt.axes, 1)):
            corner_plt.axes[i, j].set_visible(False)

        # fix decimal representation
        for ax in corner_plt.axes.flat:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))

        plt.subplots_adjust(top=0.9)
        if save:
            if save_to_thesis:
                image_path = '../Thesis/Figures/cornerplots-{}.pdf'.format(self.mol)
            else:
                image_path = self.image_outpath + '_corner.pdf'

            plt.savefig(image_path)
            print('Image saved image to ' + image_path)
        else:
            plt.show()


    def corner_dfm(self, save_to_thesis=False):
        """Make a corner plot using Dan Foreman-Mackey's Corner package."""

        # Not all the machines have this installed, so only load it if necessary.
        import corner

        plt.close()

        # Get best_fit and posterior statistics
        stats = self.groomed.describe(percentiles=[0.16, 0.84]).drop([
            'count', 'min', 'max', 'mean'])
        stats.loc['best fit'] = self.main.loc[self.main['lnprob'].idxmax()]
        stats = stats.iloc[[-1]].append(stats.iloc[:-1])
        stats.loc[['16%', '84%'], :] -= stats.loc['50%', :]
        stats = stats.reindex(
            ['50%', '16%', '84%', 'best fit', 'std'], copy=False)
        print((stats.T.round(6).to_string()))

        groomed = self.groomed.reset_index()

        p = groomed[groomed['lnprob'] == max(groomed['lnprob'])]
        bf_idx = p.index.values[0]
        bestfit = groomed.iloc[bf_idx].tolist()

        # groomed = groomed.drop(['index', 'lnprob'], axis=1)


        not_diskA_params = [p for p in groomed.columns if '_A' not in p]
        not_diskB_params = [p for p in groomed.columns if '_B' not in p]

        groomed_diskA = groomed.drop(not_diskA_params, axis=1)
        groomed_diskB = groomed.drop(not_diskB_params, axis=1)


        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # for ax, df, disk_ID in zip(axes, [groomed_diskA, groomed_diskB], ('A', 'B')):
        if save_to_thesis:
            outpath = '/Volumes/disks/jonas/Thesis/Figures/'

        else:
            outpath = self.image_outpath + '_'
        for df, disk_ID in zip([groomed_diskA, groomed_diskB], ['A', 'B']):
            print("Making corner plot")
            corner.corner(df, quantiles=[0.16,0.5,0.84], verbose=False,
                          #, labels=labels,title_args={'fontsize': 12})
                          show_titles=True, truths=bestfit)

            plt.savefig('{}cornerplot-{}-disk{}.png'.format(outpath, self.mol, disk_ID), dpi=200)
            print("Saved plot for disk{} to {}cornerplot-{}-disk{}.png".format(outpath, self.mol, disk_ID))

        # corner.corner(groomed_diskA, quantiles=[0.16,0.5,0.84], verbose=False, show_titles=True, truths=bestfit)#, labels=labels,title_args={'fontsize': 12})

        # corner.corner(groomed_diskB, quantiles=[0.16,0.5,0.84], verbose=False, show_titles=True, truths=bestfit)#, labels=labels,title_args={'fontsize': 12})


        # plt.savefig(self.image_outpath + '_corner.pdf')
        # print "Saved plot."


    # This needs multi-line updating
    def make_best_fits(self, plot_bf=True):
        """Make a best-fit model fits file."""

        bf_disk_params = list(self.bf_param_dict.keys())

        print('Making model...')

        def make_model(param_dict, mol, fourlinefit_tf=False):
            # # If we're doing four-line fitting, then some values are dictionaries of vals.
            # # Just want the relevant line.
            # param_dict_mol = param_dict.copy()
            # for p in param_dict_mol:
            #     if type(param_dict_mol[p]) == dict:
            #         param_dict_mol[p] = param_dict_mol[p][mol]

            obs = fitting.Observation(mol, cut_baselines=True)
            model = fitting.Model(observation=obs,
                                  run_name=self.name,
                                  model_name=self.name + '_bestFit')
            run_driver.make_fits(model, self.bf_param_dict, mol)
            # Make the normal map, then make a residual map.
            tools.sample_model_in_uvplane(model.modelfiles_path, mol=mol, option='replace')
            tools.sample_model_in_uvplane(model.modelfiles_path, mol=mol, option='subtract')

            # Maybe get rid of stuff in the way.
            tools.icr(model.modelfiles_path, mol=mol)
            tools.icr(model.modelfiles_path + '_resid', mol=mol)

            if plot_bf:
                tools.plot_fits(model.modelfiles_path + '.fits', mol=mol,
                                   best_fit=True, save=True)
                tools.plot_fits(model.modelfiles_path + '_resid.fits', mol=mol,
                                   best_fit=True, save=True)
            return model

        models = []
        # If it's a one line fit, it's easy.
        fourlinefit_tf = False
        if not fourlinefit_tf:
            models.append(make_model(self.bf_param_dict, self.bf_param_dict['mol']))

        else:
            bf_param_dict = self.bf_param_dicts
            # This assumes that outer radius and abundance are the only things
            # being individually varied. Maybe coordinate better.
            for m in ['cs', 'co', 'hco', 'hcn']:
                # Doesn't matter that we've still got the other lines' info here
                bf_param_dict['r_out_A'] = float(bf_param_dict['r_out_A-{}'.format(m)])
                bf_param_dict['r_out_B'] = float(bf_param_dict['r_out_B-{}'.format(m)])
                bf_param_dict['mol_abundance_A'] = float(bf_param_dict['mol_abundance_A-{}'.format(m)])
                bf_param_dict['mol_abundance_B'] = float(bf_param_dict['mol_abundance_B-{}'.format(m)])

                models.append(make_model(bf_param_dict, m))



        return (models, self.bf_param_dict)


    # This needs multi-line updating
    def plot_structure(self, zmax=150, cmap='inferno', save=False, save_to_thesis=False):
        """
        Plot temperature and density structure of the disk.
        Drawn from Kevin's code.
        """

        # Generate the Disk objects (from Kevin's code)
        # Can be nice (in testing) to manually insert these and not regen them every time.
        if not hasattr(self, 'diskA'):
            self.get_disk_objects(set_face_on=True)
        d1, d2 = self.diskA, self.diskB


        param_dict = self.bf_param_dict
        rmax_a, rmax_b = self.bf_param_dict['r_out_A'], self.bf_param_dict['r_out_B']
        rmax = max(rmax_a, rmax_b)

        # fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True,
        #                                  gridspec_kw = {'width_ratios':[rmax_a, rmax_b]}) # , sharex=True)
        fig, ((cbar_axes), (im_axes)) = plt.subplots(2, 2, figsize=(18, 5), # sharex=True,
                                                     gridspec_kw={'height_ratios':[1,9],
                                                                  # 'width_ratios':[rmax_a/rmax, rmax_b/rmax],
                                                                  'width_ratios':[1, 1]})
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)

        # full_fig = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 9]) #, figsize=(12, 6))
        # # full_fig.set_figheight(6)
        # # full_fig.set_figwidth(15)
        # cbar_axes = (plt.subplot(full_fig[0], figsize=(6, 0.5)), plt.subplot(full_fig[1]), figsize=(6, 0.5))
        # im_axes = (plt.subplot(full_fig[2], figsize=(6, 4)), plt.subplot(full_fig[3]), figsize=(6, 4))
        # im_axes = plt.subplots((full_fig[2], full_fig[3]))
        # ax1 = plt.subplot(gs[0])



        # fig = plt.figure(figsize=(10, 5), sharey=True)
        # fig.subplots_adjust(wspace=0.0, hspace=0.2)
        # plt.rc('axes',lw=2)

        lab_locs = [[(275, 25), (325, 50), (375, 80)],
                    [(154, -14.4), (200, -35.4), (261, -83)]]
        for i in range(2):
            d = [d1, d2][i]
            manual_locations=[(300,30),(250,60),(180,50),
                              (180,70),(110,60),(45,30)]

            manual_locations=[(300,30),(250,60),(180,50)]
            dens_contours = im_axes[i].contourf(d.r[0,:,:]/Disk.AU, d.Z[0,:,:]/d.AU,
                                                np.log10((d.rhoG/d.Xmol)[0,:,:]),
                                                np.arange(2, 15, 0.25),
                                                cmap=cmap) #, levels=50)

            col_dens_contours = im_axes[i].contour(d.r[0,:,:]/d.AU, d.Z[0,:,:]/d.AU,
                                                   np.log10(d.sig_col[0,:,:]), (-3, -2,-1),
                                                   linestyles=':', #linewidths=3,
                                                   colors='k', label='Column Density')

            temp_contours = im_axes[i].contour(d.r[0,:,:]/Disk.AU, d.Z[0,:,:]/Disk.AU,
                                               d.T[0,:,:], (50, 100, 150), #(20, 40, 60, 80, 100, 120),
                                               colors='grey', linestyles='--',
                                               label='Temperature')

            # im_axes[i].clabel(col_dens_contours, fmt='%1i', manual=lab_locs[i])

            # This is kinda janky, but whatever. Need colorbar set to the wider ranged one.
            # cb = plt.colorbar(dens_contours, label=r"$\log{ N(H_2)}$") #, cax=cbaxes) #, orientation='horizontal')
            # dens_contours.set_clim(0, 20)
            cbar = plt.colorbar(dens_contours, cax=cbar_axes[i],
                                orientation='horizontal', label=r"$\log{ N(H_2)}$")
            # plt.clim(-1, 1)

        font_cb = matplotlib.font_manager.FontProperties(family='times new roman',
                                                         style='italic', size=16) #, rotation=180)
        # cb.ax.yaxis.label.set_font_properties(font_cb)
        # cb.ax.yaxis.set_ticks_position('left')

        #plt.colorbar(cs2,label='log $\Sigma_{FUV}$')
        im_axes[0].set_ylabel('Scale Height (au)', fontsize=18) #, weight='bold')
        im_axes[0].set_xlabel('d253-1536a Radius (au)', fontsize=18) #, weight='bold')
        im_axes[1].set_xlabel('d253-1536b Radius (au)', fontsize=18) #, weight='bold')


        zmin = -zmax
        zmin = 0
        im_axes[0].set_xlim(rmax_a, 0)
        im_axes[0].set_ylim(zmin, zmax)
        im_axes[1].set_xlim(0, rmax_b)
        im_axes[1].set_ylim(zmin, zmax)
        im_axes[0].yaxis.set_ticks_position('left')
        im_axes[1].yaxis.set_ticks_position('right')

        freq = str(round(lines[self.mol]['restfreq'], 2)) + ' GHz'
        trans = '({}-{})'.format(lines[self.mol]['jnum'] + 1, lines[self.mol]['jnum'])
        mol = r"HCO$^+$" if self.mol.lower() == 'hco' else self.mol.upper()

        im_axes[0].text(0.22 * rmax, 0.9 * zmax, mol + trans, color='white',
                        fontsize=18, weight='bold')

        im_axes[0].text(0.22 * rmax, 0.82 * zmax, freq, color='white',
                        fontsize=14, weight='bold')


        dark_cm = matplotlib.cm.get_cmap(cmap)(0.)
        im_axes[0].set_facecolor(dark_cm)
        im_axes[1].set_facecolor(dark_cm)


        # cbar_axes[0].set_title('Disk A', weight='bold')
        # cbar_axes[1].set_title('Disk B', weight='bold')
        cbar_axes[0].xaxis.set_ticks_position('top')
        cbar_axes[1].xaxis.set_ticks_position('top')

        cbar_axes[0].xaxis.set_label_position('top')
        cbar_axes[1].xaxis.set_label_position('top')

        im_axes[1].legend()
        # fig.suptitle('d253-1536ab Temperature and Density Structures ({})'.format(mol), weight='bold', fontsize=20)

        if save:
            if save_to_thesis:
                outname = '../Thesis/Figures/diskstructures-{}.pdf'.format(self.mol)
            else:
                outname = self.image_outpath + '_disk-strs.pdf'

            plt.savefig(outname)
            print("Saved to {}".format(outname))
        else:
            fig.show()


    def plot_chi2(self, save=False):
        # Very incomplete
        chi2s = self.groomed
        return None


    # This needs multi-line updating
    def DMR_images(self, cmap='Reds', save=False, save_to_thesis=False):
        """
        Plot a triptych of data, model, and residuals.

        It would be nice to have an option to also plot the grid search results.
        Still need to:
        - Get the beam to plot
        - Get the velocity labels in the right places

        Some nice cmaps: magma, rainbow
        """
        # plt.close()
        print("\nPlotting DMR images...")


        data_path  = self.data_path + '.fits'
        resid_path = self.modelfiles_path + '_bestFit_resid.fits'
        model_path = self.modelfiles_path + '_bestFit.fits'

        # if not Path(model_path).exists():
        #     print("No best-fit model made yet; making now...")

        # tools.remove(self.modelfiles_path + '_bestFit*')
        # self.make_best_fits(plot_bf=False)

        real_data    = fits.getdata(data_path, ext=0).squeeze()
        model_data   = fits.getdata(model_path, ext=0).squeeze()
        resid_data   = fits.getdata(resid_path, ext=0).squeeze()
        image_header = fits.getheader(data_path, ext=0)
        model_header = fits.getheader(model_path, ext=0)
        resid_header = fits.getheader(resid_path, ext=0)


        # Set up some physical params
        # vmin, vmax = np.nanmin(real_data), np.nanmax(real_data)
        vmax = np.nanmax((np.nanmax(real_data), -np.nanmin(real_data)))
        vmin, vmax = np.nanmin(real_data), np.nanmax(real_data)
        # vmin = -vmax

        offsets = self.param_dict['offsets']
        offsets_dA, offsets_dB = offsets[0], offsets[1]
        x_center = int(np.floor(real_data.shape[1] / 2))
        crop_arcsec = 2
        crop_pix = int(crop_arcsec / 0.045)
        xmin, xmax = x_center - crop_pix, x_center + crop_pix
        offsets_dA_pix = [44 - offsets_dA[0]/0.045, 44 - offsets_dA[1]/0.045]
        offsets_dB_pix = [44 - offsets_dB[0]/0.045, 44 + offsets_dB[1]/0.045]

        chanstep_vel = image_header['CDELT3'] * 0.001
        chan0_vel = image_header['CRVAL3'] * 0.001 - image_header['CRPIX3'] * chanstep_vel

        ch_offsets = {'co':2, 'hco':17, 'hcn': 20, 'cs': 10}
        chan_offset = ch_offsets[self.mol]
        nchans = 24

        rms = imstat(data_path[:-5])[1]
        contours = [rms * i for i in range(3, 30, 3)]


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
        n_rows = 4
        n_cols = int(np.ceil(nchans/n_rows))

        # Get the plots going
        # fig = plt.figure(figsize=(n_rows * 3, 7))
        fig = plt.figure(figsize=(8.5, 11))
        fig.subplots_adjust(wspace=0., hspace=0.13, top=0.95, left=0.07, right=0.88)
        cax = plt.axes([0.9, 0.1, 0.02, 0.85]) # [l, b, w, h]

        big_fig = gridspec.GridSpec(3, 1)

        # Define some plotting params
        hspace, wspace = 0.05, 0.05
        # hspace, wspace = 0, 0
        data_ims = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                                                    wspace=wspace, hspace=hspace,
                                                    subplot_spec=big_fig[0])
        model_ims = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                                                     wspace=wspace, hspace=hspace,
                                                     subplot_spec=big_fig[1])
        resid_ims = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                                                     wspace=wspace, hspace=hspace,
                                                     subplot_spec=big_fig[2])
        # Populate the plots
        print("Got the necessary info; now plotting...")
        for i in range(nchans):
            chan = i + chan_offset
            velocity = str(round(chan0_vel + chan * chanstep_vel, 2))
            print ("Plotting {} km/s".format(velocity))
            ax_d = plt.Subplot(fig, data_ims[i])
            ax_m = plt.Subplot(fig, model_ims[i])
            ax_r = plt.Subplot(fig, resid_ims[i])

            if i == int(np.floor(n_cols / 2)):
                ax_d.set_title('Data', weight='bold')
                ax_m.set_title('Model', weight='bold')
                ax_r.set_title('Residuals', weight='bold')

            # Plot the data
            # REWORK contours all black
            im_d = ax_d.contourf(real_data[i + chan_offset][xmin:xmax, xmin:xmax],
                                 # extent=(crop_pix, -crop_pix, crop_pix, -crop_pix),
                                 levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
            im_m = ax_m.contourf(model_data[i + chan_offset][xmin:xmax, xmin:xmax],
                                 # extent=(crop_pix, -crop_pix, crop_pix, -crop_pix),
                                 levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
            im_r = ax_r.contourf(resid_data[i + chan_offset][xmin:xmax, xmin:xmax],
                                 # extent=(crop_pix, -crop_pix, crop_pix, -crop_pix),
                                 levels=30, cmap=cmap, vmin=vmin, vmax=vmax)


            # Add n-sigma contours
            im_d = ax_d.contour(real_data[i + chan_offset][xmin:xmax, xmin:xmax],
                                 # extent=(crop_pix, -crop_pix, crop_pix, -crop_pix),
                                 levels=contours, colors='k', alpha=0.7, linewidths=0.3)
            im_m = ax_m.contour(model_data[i + chan_offset][xmin:xmax, xmin:xmax],
                                 # extent=(crop_pix, -crop_pix, crop_pix, -crop_pix),
                                 levels=contours, colors='k', alpha=0.7, linewidths=0.3)
            im_r = ax_r.contour(resid_data[i + chan_offset][xmin:xmax, xmin:xmax],
                                 # extent=(crop_pix, -crop_pix, crop_pix, -crop_pix),
                                 levels=contours, colors='k', alpha=0.7, linewidths=0.3)




            # Aesthetic stuff
            # This is all in arcsecs right now. Should be in pix
            # crop_arcsec of 2 translates to 88 pixels across
            # 0, 0 in upper left
            ax_d.grid(False)
            ax_d.xaxis.set_ticks([]), ax_d.yaxis.set_ticks([])
            ax_d.set_xticklabels([]), ax_d.set_yticklabels([])
            ax_d.plot(offsets_dA_pix[0], offsets_dA_pix[1], '+g')
            ax_d.plot(offsets_dB_pix[0], offsets_dB_pix[1], '+g')

            ax_m.grid(False)
            ax_m.xaxis.set_ticks([]), ax_m.yaxis.set_ticks([])
            ax_m.set_xticklabels([]), ax_m.set_yticklabels([])
            ax_m.plot(offsets_dA_pix[0], offsets_dA_pix[1], '+g')
            ax_m.plot(offsets_dB_pix[0], offsets_dB_pix[1], '+g')

            ax_r.grid(False)
            # ax_r.xaxis.set_ticks([]), ax_r.yaxis.set_ticks([])
            # ax_r.xaxis.set_ticks([]), ax_r.yaxis.set_ticks([])
            # ax_r.set_xticklabels([]), ax_r.set_yticklabels([])
            ax_r.plot(offsets_dA_pix[0], offsets_dA_pix[1], '+g')
            ax_r.plot(offsets_dB_pix[0], offsets_dB_pix[1], '+g')

            # Add velocity info
            ax_d.text(44, 75, velocity + ' km/s', fontsize=12, color='k',
                    horizontalalignment='center', verticalalignment='center')
            ax_m.text(44, 75, velocity + ' km/s', fontsize=12, color='k',
                    horizontalalignment='center', verticalalignment='center')
            ax_r.text(44, 75, velocity + ' km/s', fontsize=12, color='k',
                    horizontalalignment='center', verticalalignment='center')

            # If we're in the bottom left corner, add a beam and axis labels
            if i == n_cols * (n_rows - 1) and add_beam_d is True:
                el = Ellipse(xy=(20, 20),
                             # xy=(0.8 * crop_arcsec, 0.8 * crop_pix),
                             width=bmin/0.045, height=bmaj/0.045, angle=-bpa,
                             fc='k', ec='k', fill=False, hatch='////////')
                ax_r.add_artist(el)
                print("Beam properties: {}, {}, {}".format(bmin, bmaj, bpa))

                # return ax_r
                tickmin, tickmax = ax_r.get_xlim()
                tickmed = (tickmin + tickmax) * 0.045/2
                # ticks = [i * 0.045 - tickmed for i in np.linspace(tickmin, tickmax, 7)]
                ticks = np.linspace(tickmin, tickmax, 7)

                ax_r.xaxis.set_ticks(ticks), ax_r.yaxis.set_ticks(ticks)
                ax_r.tick_params(axis='both', direction='inout') #, which='both')

                ax_r.set_xticklabels(['', 1, '', 0, '', -1, '']), ax_r.set_yticklabels(['', -1, '', 0, '', 1, ''])

                ax_r.set_xlabel(r"$\Delta \alpha$", weight='bold'), ax_r.set_ylabel(r"$\Delta \delta$", weight='bold')
            else:
                ax_r.xaxis.set_ticks([]), ax_r.yaxis.set_ticks([])
                ax_r.set_xticklabels([]), ax_r.set_yticklabels([])



            fig.add_subplot(ax_m)
            fig.add_subplot(ax_d)
            fig.add_subplot(ax_r)



        # fig.tight_layout()

        cmaps = plt.imshow(real_data[i + chan_offset][xmin:xmax, xmin:xmax],
                           extent=(crop_arcsec, -crop_arcsec, crop_arcsec, -crop_arcsec),
                           cmap=cmap, vmin=vmin, vmax=vmax)

        cbar = plt.colorbar(cmaps, cax=cax, orientation='vertical')
        cbar.set_label('Jy/beam', labelpad=-0, fontsize=20, weight='bold', rotation=270)
        # REWORK: Better ticks
        cbar_ticks_raw = [round(i, 2) for i in np.linspace(vmin, vmax, 6)]
        # cbar_ticks = cbar_ticks_raw[:2] + cbar_ticks_raw[3:]
        cbar.set_ticks(cbar_ticks_raw)



        out_path = self.image_outpath + '_DMR-images.png'
        thesis_fig_path = '../Thesis/Figures/DMRchanmaps_{}.pdf'.format(self.mol)
        if save:
            if save_to_thesis:
                plt.savefig(thesis_fig_path)
                print("Saved to " + thesis_fig_path)
            else:
                plt.savefig(out_path, dpi=300)
                print("Saved to " + out_path)
        else:
            print("Showing")
            plt.show(block=False)


    # This needs multi-line updating
    def plot_pv_diagram(self, coords=None, save=False):
        """
        Fuck Miriad and CASA, let's just use a package.

        Args: image_path (str): path to fits image, including extension.
              coords (tuple of tuples): if you have x and y values for the
                                        disk axis, enter them.
        """

        image_path = self.data_path + '.fits'
        # Not all the machines have this package installed,
        # so don't run it unless necessary.
        from pvextractor import extract_pv_slice
        from pvextractor import Path as PVPath
        # Can use this to test for points:
        if coords is None:
            keep_trying = True
            # For HCN
            xs, ys = [38, 57], [55, 45]
            xs, ys = [110, 142], [135, 124]

            while keep_trying:
                plt.close()
                print("Find coordinates for a line across the disk axis:")

                # Import and crop the data, 70 pixels in each direction.
                image_data_3d = fits.getdata(image_path).squeeze() #[:, 80:176, 80:176]
                # Make a quick moment map
                image_data = np.sum(image_data_3d, axis=0)

                # Plot
                plt.contourf(image_data, 50, cmap='BrBG')
                plt.colorbar(extend='both')
                plt.contour(image_data, colors='k', linewidths=0.2)
                plt.plot(xs, ys, '-k')
                plt.show(block=False)
                response = input('\nWant to try again?\n[y/n]: ').lower()
                keep_trying = True if response == 'y' or response == 'yes' else False
                if keep_trying:
                    xs_raw = input('Enter the x coordinates (previous attempt: {}):\n[x1, x2]: '
                                       .format(xs))
                    xs = tuple(int(x.strip()) for x in xs_raw.split(','))

                    ys_raw = input('Enter the x coordinates (previous attempt: {}):\n[y1, y2]: '
                                       .format(ys))
                    ys = tuple(int(x.strip()) for x in ys_raw.split(','))
        else:
            xs, ys = coords


        path = PVPath([(xs[0], ys[0]), (xs[1], ys[1])])
        pv_data = extract_pv_slice(image_path, path).data.T
        # pv_data = extract_pv_slice(image_data_3d, path).data.T


        # Make the plot.
        plt.close()
        plt.clf()
        fig, (ax_image, ax_pv) = plt.subplots(1, 2, figsize=(10, 5),
                                              gridspec_kw={'width_ratios':[2, 2]})

        ax_image.contourf(image_data, 50, cmap='BrBG')
        #   ax_image.colorbar(extend='both')
        ax_image.contour(image_data, colors='k', linewidths=0.2)
        ax_image.plot(xs, ys, '-k')

        ax_pv.contourf(pv_data, 30, cmap='inferno')
        # ax_pv.colorbar(extend='both')
        ax_pv.contour(pv_data, 30, colors='k', linewidths=0.1)


        # Image aesthetics
        pixel_to_AU = 0.045 * 389   # arcsec/pixel * distance -> AU
        pixel_to_as = 0.045
        pv_ts = np.array(ax_pv.get_xticks().tolist()) * pixel_to_AU

        xmin, xmax = ax_pv.get_xlim()
        pv_tick_labels = (np.linspace(xmin, xmax, 5) - np.mean([xmin, xmax])) * pixel_to_AU
        pv_tick_labels = [int(tick) for tick in pv_tick_labels]

        ax_pv.xaxis.set_xticks(np.linspace(xmin, xmax, 5))
        ax_pv.set_xticklabels(pv_tick_labels)



        vmin, vmax = ax_pv.get_ylim()
        vel_tick_labels = np.linspace(vmin, vmax, 5) - np.mean([vmin, vmax])
        vel_tick_labels = [int(tick) for tick in vel_tick_labels]

        ax_pv.yaxis.set_ticks(np.linspace(vmin, vmax, 5))
        ax_pv.set_yticklabels(vel_tick_labels)

        mol = self.mol
        plt.suptitle('Moment Map and PV Diagram for {}'.format(mol), weight='bold')
        # plt.tight_layout()

        outpath = self.image_outpath + '_pv-diagram'
        if save:
            plt.savefig(outpath + '.pdf')
            print("Saved PV diagram to {}.pdf".format(outpath))
        else:
            print("Showing:")
            plt.show(block=False)

# ex_run = MCMC_Analysis('june11-multi')
# ex_run.plot_structure(save=False)




class Figure:
    """
    Make publication-quality plots:
    - zeroth- and first-moment maps
    - Disk structure

    Note that there are some big assumptions about file names/structures made here:
    1. Images are saved to ../Thesis/Figures/. If this doesn't exist, then trouble.
    2. Assumes that the name of the molecular line of the observation is in the
        fits file name. Without this (or with conflicting ones), it won't be able
        to determine which line a fits file is represents.

    Some nice colormaps:
        RdBu, cividis, Spectral
    """

    # Set seaborn plot styles and color pallete
    sns.set_style("ticks", {"xtick.direction": "in", "ytick.direction": "in"})
    sns.set_context("paper")

    def __init__(self, paths, make_plot=True, save=False, moment=0, remove_bg=True,
                 texts=None, title=None, image_outpath=None, export_fits_mom=False,
                 plot_bf_ellipses=False, cmap='RdBu'):
        """
        Make a nice image from a fits file.

        Args:
            paths (list or str): paths to the fits file to be used, including .fits
            make_plot (bool): Whether or not to actually make the plots at all.
            save (bool): Whether or not to save the image.
                         If False, it will be shown instead.
            moment (0 or 1): Which moment map to make.
            remove_bg (bool): Whether or not to generate a mask to white out
                              pixels with intensities less than n*sigma
            texts (str): Idk. Thinking about getting rid of this.
            title (str): Title for the whole plot.
        """
        self.title = title
        self.moment = moment
        self.export_fits_mom = export_fits_mom
        self.remove_bg = remove_bg
        self.paths = np.array(([paths]) if type(paths) is str else paths)
        self.plot_bf_ellipses = plot_bf_ellipses
        self.cmap = cmap
        # This is gross but functional. The break is important.
        self.mols = []
        mols = ['hco', 'hcn', 'cs', 'co']
        for path in self.paths:
            for mol in mols:
                if mol in path:
                    self.mols.append(mol)
                    break


        self.outpath = '../Thesis/Figures/m{}-map_{}.png'.format(moment,
                                                                '-'.join(self.mols),
                                                                dpi=300)
        self.outpath = self.outpath if image_outpath is None else image_outpath

        # Clear any pre existing figures, then create figure
        plt.close()
        if make_plot:
            self.rows, self.columns = (1, len(self.paths))
            self.fig, self.axes = plt.subplots(self.rows, self.columns,
                                               figsize=(
                                                   # 11.6/2 * self.columns, 6.5*self.rows),
                                                   7*self.columns, 6.5*self.rows),
                                               sharex=False, sharey=True, squeeze=False)
            plt.subplots_adjust(wspace=-0.0)

            texts = np.array([texts], dtype=object) if type(texts) is str \
                else np.array(texts, dtype=object)
            # What is this doing?
            if type(texts.flatten()[0]) is not float:
                texts = texts.flatten()

            # Get global vmin and vmax
            vmin, vmax = 1e10, -1e10
            for p in self.paths:
                d = fits.getdata(p)
                vmin = np.nanmin(d) if vmin > np.nanmin(d) else vmin
                vmax = np.nanmax(d) if vmax < np.nanmax(d) else vmax
            self.vmin, self.vmax = vmin, vmax

            # Populate the stuff
            print((self.paths, self.mols))
            for ax, path, mol in zip(self.axes.flatten(), self.paths, self.mols):
                self.get_fits(path, mol)
                self.make_axis(ax)
                self.fill_axis(ax, mol)

            if save:
                # if image_outpath:
                #     plt.savefig(image_outpath, dpi=200)
                #     print("Saved image to {}.png".format(image_outpath))
                # else:

                plt.savefig(self.outpath, dpi=200)
                print("Saved image to {}".format(self.outpath))
            else:
                plt.show(block=False)


    def get_fits_manually(self, path, mol):
        """Make moment maps by hand. Should not be used."""
        fits_file = fits.open(path)
        self.head = fits_file[0].header
        self.data = fits_file[0].data.squeeze()

        # Read in header spatial info to create ra
        nx, ny, nv = self.head['NAXIS1'], self.head['NAXIS2'], self.head['NAXIS3']
        xpix, ypix = self.head['CRPIX1'], self.head['CRPIX2']
        xval, yval = self.head['CRVAL1'], self.head['CRVAL2']
        self.xdelt, self.ydelt = self.head['CDELT1'], self.head['CDELT2']

        # Convert from degrees to arcsecs
        self.ra_offset = np.array(
            ((np.arange(nx) - xpix + 1) * self.xdelt) * 3600)
        self.dec_offset = np.array(
            ((np.arange(ny) - ypix + 1) * self.ydelt) * 3600)

        # Assumes we have channels (i.e. nv > 1)
        try:
            self.rms = imstat(path.split('.')[-2])[1] * nv
        except sp.CalledProcessError:
            self.rms = 0
        # Decide which moment map to make.
        # www.atnf.csiro.au/people/Tobias.Westmeier/tools_hihelpers.php#moments
        if self.moment == 0:
            # Integrate intensity over pixels.
            self.im = np.sum(self.data, axis=0)
            if self.remove_bg:
                self.im = ma.masked_where(self.im < self.rms, self.im, copy=True)

        elif self.moment == 1:
            self.im = np.zeros((nx, ny))

            vsys = constants.obs_stuff(mol)[0]
            obsv = constants.obs_stuff(mol)[3] - vsys[0]

            # There must be a way to do this with array ops.
            # obsv = obsv.reshape([len(obsv)]).shape
            # self.im = np.sum(self.data * obsv, axis=0)/np.sum(self.data, axis=0)

            for x in range(nx):
                for y in range(ny):
                    # I think this is doing good stuff.
                    self.im[x, y] = np.sum(self.data[:, x, y] * obsv)
                    # self.im[x, y] = np.sum(self.data[:, x, y] * obsv)/np.sum(self.data[:, x, y])

        if self.remove_bg:
            self.im = ma.masked_where(abs(self.im) < self.rms, self.im, copy=True)

        if self.export_fits_mom:
            fits_out = fits.PrimaryHDU()
            fits_out.header = self.head
            fits_out.data = self.im
            modeling = '/Volumes/disks/jonas/modeling/'
            outpath = modeling + 'data/{}/{}-moment{}.fits'.format(mol, mol,
                                                                   self.moment)
            fits_out.writeto(outpath, overwrite=True)
            print(("Wrote out moment {} fits file to {}".format(self.moment,
                                                               outpath)))
            # change units to micro Jy
        # self.im *= 1e6
        # self.rms *= 1e6


    def get_fits(self, path, mol):
        """Docstring."""
        fits_file = fits.open(path)
        self.head = fits_file[0].header
        self.data = fits_file[0].data.squeeze()

        # Read in header spatial info to create ra
        nx, ny, nv = self.head['NAXIS1'], self.head['NAXIS2'], self.head['NAXIS3']
        xpix, ypix = self.head['CRPIX1'], self.head['CRPIX2']
        xval, yval = self.head['CRVAL1'], self.head['CRVAL2']
        self.xdelt, self.ydelt = self.head['CDELT1'], self.head['CDELT2']

        # Convert from degrees to arcsecs
        self.ra_offset = np.array(
            ((np.arange(nx) - xpix + 1) * self.xdelt) * 3600)
        self.dec_offset = np.array(
            ((np.arange(ny) - ypix + 1) * self.ydelt) * 3600)

        # Check if we're looking at 2- or 3-dimensional data
        if len(self.data.shape) == 3:
            # Make some moment maps. Make both maps and just choose which data to use.
            momentmap_basepath = path.split('.')[-2]
            moment_maps(momentmap_basepath, clip_val=0, moment=0)
            self.rms = imstat_single(momentmap_basepath + '.moment0')[1]

            moment_maps(momentmap_basepath, clip_val=self.rms, moment=0)
            self.im = fits.getdata(momentmap_basepath + '.moment0.fits').squeeze()

            if self.moment == 1:
                moment_maps(momentmap_basepath, clip_val=self.rms, moment=1)
                self.im_mom1 = fits.getdata(momentmap_basepath + '_moment1.fits').squeeze()

        else:
            self.im = self.data
            self.rms = imstat_single(path.split('.')[-2])[1]


        if self.export_fits_mom:
            fits_out = fits.PrimaryHDU()
            fits_out.header = self.head
            data = self.im if self.moment is 0 else self.im_mom1
            fits_out.data = data[100:160, 100:180]
            modeling = '/Volumes/disks/jonas/modeling/'
            outpath = modeling + 'data/{}/{}-moment{}.fits'.format(mol, mol,
                                                                   self.moment)
            fits_out.writeto(outpath, overwrite=True)
            print(("Wrote out moment {} fits file to {}".format(self.moment,
                                                               outpath)))
            print("NOTE: ^^ That moment map was cropped (in line ~800)")
        # change units to micro Jy
        # self.im *= 1e6
        # self.rms *= 1e6


    def make_axis(self, ax):
        """Docstring."""
        # Set seaborn plot styles and color pallete
        sns.set_style("ticks",
                      {"xtick.direction": "in",
                       "ytick.direction": "in"})
        sns.set_context("talk")

        xmin = -2.0
        xmax = 2.0
        ymin = -2.0
        ymax = 2.0
        ax.set_xlim(xmax, xmin)
        ax.set_ylim(ymin, ymax)
        ax.grid(False)

        # Set x and y major and minor tics
        majorLocator = MultipleLocator(1)
        ax.xaxis.set_major_locator(majorLocator)
        ax.yaxis.set_major_locator(majorLocator)

        minorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_minor_locator(minorLocator)

        # Set x and y labels
        ax.set_xlabel(r'$\Delta \alpha$ (")', fontsize=18)
        ax.set_ylabel(r'$\Delta \delta$ (")', fontsize=18)

        # tick_labs = ['', '', '-4', '', '-2', '', '0', '', '2', '', '4', '']
        tick_labs = ['', '', '-1', '', '1', '', '']
        ax.xaxis.set_ticklabels(tick_labs, fontsize=18)
        ax.yaxis.set_ticklabels(tick_labs, fontsize=18)
        ax.tick_params(which='both', right='on', labelsize=18, direction='in')

        # Set labels depending on position in figure
        if np.where(self.axes == ax)[1] % self.columns == 0:  # left
            ax.tick_params(axis='y', labelright='off', right='on')
        elif np.where(self.axes == ax)[1] % self.columns == self.columns - 1:  # right
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft='off', labelright='on')
        else:  # middle
            ax.tick_params(axis='y', labelleft='off')
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Set physical range of colour map
        self.extent = [self.ra_offset[0], self.ra_offset[-1],
                       self.dec_offset[-1], self.dec_offset[0]]


    def fill_axis(self, ax, mol):
        """Plot image as a colour map."""

        # This is massively hacky, but works. Basically, if we're plotting as
        # moment1 map, we still want to contour with moment0 lines. So this:
        try:
            im = self.im_mom1
            cbar_lab = r'$km \, s^{-1}$'
            vmin, vmax = 7, 13

        except AttributeError:
            im = self.im
            cbar_lab = r'$Jy / beam$'
            vmax = max((-np.nanmin(im), np.nanmax(im)))
            vmin = -vmax
            vmin, vmax = np.nanmin(im), np.nanmax(im)
            vmin, vmax = self.vmin, self.vmax

        cmap = ax.imshow(im, extent=self.extent,
                         vmin=vmin, vmax=vmax,
                         cmap=self.cmap)


        if self.rms:
            cont_levs = np.arange(3, 30, 3) * self.rms
            # add residual contours if resdiual exists; otherwise, add image contours
            try:
                ax.contour(self.resid,
                           levels=cont_levs,
                           colors='k',
                           linewidths=0.75,
                           linestyles='solid')
                ax.contour(self.resid,
                           levels=-1 * np.flip(cont_levs, axis=0),
                           colors='k',
                           linewidths=0.75,
                           linestyles='dashed')
            except AttributeError:
                ax.contour(self.ra_offset, self.dec_offset, self.im,
                           colors='k',
                           levels=cont_levs,
                           linewidths=0.75,
                           linestyles='solid')
                ax.contour(self.ra_offset, self.dec_offset, self.im,
                           levels=-1 * np.flip(cont_levs, axis=0),
                           colors='k',
                           linewidths=0.75,
                           linestyles='dashed')

        # Create the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="8%", pad=0.0)
        cbar = self.fig.colorbar(cmap, ax=ax, cax=cax, orientation='horizontal')
        cbar.ax.xaxis.set_tick_params(direction='out', length=3, which='major',
                                      bottom='off', top='on', labelsize=12, pad=-2,
                                      labeltop='on', labelbottom='off')

        cbar.ax.xaxis.set_tick_params(direction='out', length=2, which='minor',
                                      bottom='off', top='on')

        if np.nanmax(self.im) > 500:
            tickmaj, tickmin = 200, 50
        elif np.nanmax(self.im) > 200:
            tickmaj, tickmin = 100, 25
        elif np.nanmax(self.im) > 100:
            tickmaj, tickmin = 50, 10
        elif np.nanmax(self.im) <= 100:
            tickmaj, tickmin = 20, 5


        # minorLocator = AutoMinorLocator(tickmaj / tickmin)
        # cbar.ax.xaxis.set_minor_locator(minorLocator)
        # cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(),
        #                         rotation=45, fontsize=18)
        # cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), fontsize=18)
        # cbar.set_ticks(np.arange(-10*tickmaj, 10*tickmaj, tickmaj))

        # Colorbar label. No idea why the location of this is so weird.
        # cbar.ax.text(0.425, 0.320, r'$\mu Jy / beam$', fontsize=12,
        # cbar.ax.text(0.425, 0.320, cbar_lab, fontsize=12,

        cbar_x, cbar_y = np.mean((vmin, vmax)), np.mean((vmin, vmax))
        cbar.ax.text(cbar_x, cbar_y, cbar_lab, fontsize=12, ha='center', va='center',
                     path_effects=[PathEffects.withStroke(linewidth=2, foreground="w")])
        # cbar.set_label(cbar_lab, fontsize=13, path_effects=[PathEffects.withStroke(linewidth=4, foreground="w")],
        #                labelpad=-5)


        # Overplot the beam ellipse
        try:
            bmin = self.head['bmin'] * 3600.
            bmaj = self.head['bmaj'] * 3600.
            bpa = self.head['bpa']

            el = Ellipse(xy=[1.5, -1.5], width=bmin, height=bmaj, angle=-bpa,
                         edgecolor='k', hatch='///', facecolor='white', zorder=10)
            ax.add_artist(el)
        except KeyError:
            print("Unable to plot beam; couldn't find header info.")

        # Plot the scale bar
        if np.where(self.axes == ax)[1][0] == 0:  # if first plot
            x, y = -0.8, -1.7        # arcsec location
            ax.plot([x, x - 400/389], [y, y], '-', linewidth=3, color='darkorange')
            ax.text(x - 0.1, y + 0.15, "400 au", fontsize=12,
                path_effects=[PathEffects.withStroke(linewidth=2, foreground="w")])

        # Plot crosses at the source positions
        (posx_A, posy_A), (posx_B, posy_B) = constants.offsets
        ax.plot([posx_A], [posy_A], '+', markersize=6,
                markeredgewidth=2, color='darkorange')
        ax.plot([posx_B], [posy_B], '+', markersize=6,
                markeredgewidth=2, color='darkorange')

        # Plot ellipses with each disk's best-fit radius and inclination:
        print('\n\n\n\nLine is {}\n\n\n\n\n'.format(mol))
        # if mol.lower() == 'hcn':
        if self.plot_bf_ellipses is True:
            print("\n\n\nAdding ellipses")
            # ax.set_xlim(-1, 2)
            # ax.set_ylim(-1, 2)
            r_A = (340/389)
            r_B1 = (380/389)
            r_B2 = (150/389)
            PA_A, PA_B = 90 - 69, 136
            incl_A, incl_B = 65, 45
            print("Note that this is manually HCN specific rn, with:")
            print("rA = {}\nrB_out = {}\nrB_in = {}\nand some PAs/incls\n\n\n\n".format(r_A, r_B1, r_B2))
            ellipse_A = Ellipse(xy=(posx_A, posy_A),
                                width=r_A, height=r_A*np.sin(incl_A), angle=PA_A,
                                fill=False, edgecolor='orange', ls='-', lw=5, label='R = 334 AU')
            ellipse_B1 = Ellipse(xy=(posx_B, posy_B),
                                 width=r_B1, height=r_B1*np.sin(incl_B), angle=PA_B,
                                 fill=False, edgecolor='orange', ls='-', lw=5, label='R = 324 AU')
            ellipse_B2 = Ellipse(xy=(posx_B, posy_B),
                                 width=r_B2, height=r_B2*np.sin(incl_B), angle=PA_B,
                                 fill=False, edgecolor='r', ls='-', lw=5, label='R = 145 AU')
            ax.add_artist(ellipse_A)
            ax.add_artist(ellipse_B1)
            ax.add_artist(ellipse_B2)

        # Annotate with some text:
        freq = str(round(lines[mol]['restfreq'], 2)) + ' GHz'
        trans = '({}-{})'.format(lines[mol]['jnum'] + 1, lines[mol]['jnum'])
        molname = r'HCO$^+$(4-3)' if mol is 'hco' else mol.upper() + trans
        sysname = 'd253-1536'

        # Print the system name.
        ax.text(1.8, 1.6, sysname,
                fontsize=20, weight='bold', horizontalalignment='left',
                path_effects=[PathEffects.withStroke(linewidth=2,
                                                     foreground="w")])

        ax.text(-1.85, 1.7, molname,
                fontsize=13, weight='bold', horizontalalignment='right',
                path_effects=[PathEffects.withStroke(linewidth=1,
                                                     foreground="w")])

        ax.text(-1.8, 1.5, freq,
                fontsize=13, horizontalalignment='right',
                path_effects=[PathEffects.withStroke(linewidth=1,
                                                     foreground="w")])

        # Add figure text
        # if text is not None:
        #     for t in text:
        #         ax.text(1, 1, *t, fontsize=18,
        #                 path_effects=[PathEffects.withStroke(linewidth=3,
        #                                                      foreground="w")])
        if self.title:
            plt.suptitle(self.title, weight='bold')

# ex_fig = Figure(['data/hco/hco.fits', 'data/hco/hco-short110.fits'], moment=1, remove_bg=True, save=True)



class Observation_Analysis():
    """
    Some plotting tools for the raw data.

    This could actually be really generally useful, beyond this project. What do we need?
    - Filetype inputs: assume fits and uvf exist?
    -

    Right now, this is just stolen sippets from other scripts. Hopefully it'll
    be cleaned up at some point.
    """

    def __init__(self, datapath):
        self.datapath = datapath


    def moment_maps():
        return None

    def channel_maps():
        return None

    def uv_noise():
        return None


    def plot_pv_diagram(fits_path): #, center=[129, 130], length=25, pa=70):
        """
        Make a position-velocity diagram with Casa
        https://casa.nrao.edu/casadocs/casa-5.1.0/global-task-list/task_impv/about

        Use imview to imview a .cm image. Click the tiny "P/V" icon under the
        toolbar (in the upper left), manually give a line that is approximately
        the disk's axis of rotation, and save out as a fits. This script just
        makes it a pretty picture.

        Note: in tools.py, we've still got a function that pipes the stuff that
        this one has you do by hand into CASA. It would be way better to use that,
        but it leads to weird images.
        """


        #image_path = './pv_diagrams/pvd_casa_byhand_hco{}.fits'.format(diskID.upper())
        d = fits.getdata(image_path).squeeze()
        vmax = max(np.nanmax(d), -np.nanmin(d))

        rms = imstat('data/hco/hco-short110')[1]
        levs = [rms * i for i in np.linspace(3, 30, 3)]
        fig, (im_ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios':[12, 1]})
        im = im_ax.contourf(d, levels=24, cmap='Spectral_r', vmin=-vmax, vmax=vmax)
        im_ax.contour(d, levels=16, colors='black', linewidths=0.3) #, vmin=-vmax, vmax=vmax)

        xmin, xmax = im_ax.get_xlim()
        im_ax.xaxis.set_ticks(np.linspace(xmin, xmax, 7))
        im_ax.set_xticklabels(['', 1, '', 0, '', -1, ''])

        vmin, vmax = im_ax.get_ylim()
        vel_tick_labels = np.linspace(vmin, vmax, 5) - np.mean([vmin, vmax])
        vel_tick_labels = [int(tick) for tick in vel_tick_labels]

        im_ax.yaxis.set_ticks(np.linspace(vmin, vmax, 5))
        im_ax.set_yticklabels(vel_tick_labels)


        cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.set_ticks(np.linspace(np.nanmin(d), np.nanmax(d), 4))
        raw_cbar_ticklabs = np.linspace(np.nanmin(d), np.nanmax(d), 4) * 1000
        cbar.set_ticklabels([round(i, 0) for i in raw_cbar_ticklabs])
        cbar.set_label('mJy/beam', labelpad=-10, fontsize=20, weight='bold', rotation=270)
        im_ax.set_ylabel("Velocity (km/s)", weight='bold') #, rotation=270)
        im_ax.set_xlabel("Position Offset (arcsec)", weight='bold')


        fig.tight_layout(w_pad=0.05)
        fig.show()

        return im_ax







class GridSearch_Analysis:
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
        print((self.mol, lines[self.mol]['baseline_cutoff']))
        self.run_date = path.split('/')[-1].split('_')[0]
        self.out_path = './gridsearch_results/{}-{}'.format(self.run_date, self.mol)
        self.data_path = './data/{}/{}-short{}.fits'.format(self.mol, self.mol,
                                                            str(lines[self.mol]['baseline_cutoff']))

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


    def plot_step_duration(self, ns=[10, 20, 50], save=False):
        """Plot how long each step took, plus some smoothing stuff.

        Args:
            ns (list of ints): A list of the smoothing windows to use.
                               Note len(ns) can't be longer than 5 without adding
                               more colors to colors list.
        """
        plt.close()
        print("\nPlotting step durations...")

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
            print(("Saved to " + self.out_path + '_durations.pdf'))
        else:
            plt.show()
        plt.clf()


    def plot_best_fit_params(self, save=False):
        """
        Plot where the best-fit values from a grid search fall.

        Plot where the best-fit value(s) stand(s) relative to the range queried in
        a given grid search run.

        Args:
            fname (str): Name of the pickled step log from the grid search.
            Assumes fname is './models/dateofrun/dateofrun'
        """
        plt.close()
        print("\nPlotting best-fit param number lines...")

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
            print(("Saved to " + self.out_path + '_bestfit_params.pdf'))
        else:
            plt.show()


    def DMR_images(self, cmap='seismic', save=False):
        """Plot a triptych of data, model, and residuals.

        It would be nice to have an option to also plot the grid search results.
        Still need to:
        - Get the beam to plot
        - Get the velocity labels in the right places

        Some nice cmaps: magma, rainbow
        """
        plt.close()
        print("\nPlotting DMR images...")

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
        print("Got the necessary info; now plotting...")
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
                el = Ellipse(xy=[0.8 * crop_arcsec, 0.8 * crop_pix],
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
            print(("Saved to " + out_path))
        else:
            print("Showing")
            fig.show()


    def DMR_spectra(self, save=False):
        """
        Plot a model/data/resid triptych of spectra
        y-axis units: each pixel is in Jy/beam, so want to:
            - Multiply each by beam
            - Divide by number of pix (x*y)?
        """
        plt.close()
        print("\nPlotting DMR spectra...")

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

            fig(self.out_path + '_DMR-spectra.pdf')
            print(("Saved to " + self.out_path + '_DMR-spectra.pdf'))
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
                    print((str([i]), p, '(Length: ' + str(len_p) + ')'))

        p1_idx = int(eval(input('Select the index of the first parameter.\n')))
        p2_idx = int(eval(input('Select the index of the second parameter.\n')))
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
        print((len_p1, len_p2))
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
                # print this_chi
                # print p_i, df[p_i][i * len_p_j + j], '; ', p_j, df[p_j][i * len_p_j + j]
                # print

        plt.close()
        # vmin, vmax = np.nanmin(df_full['Raw Chi2']), np.nanmax(df_full['Raw Chi2'])
        plt.matshow(mat, cmap='jet')  #, vmin=vmin, vmax=vmax)
        plt.xlabel(df[p_j].name)
        plt.ylabel(df[p_i].name)
        plt.xticks(list(range(len_p2)), sorted(list(set(df[param2]))))
        plt.yticks(list(range(len_p1)), sorted(list(set(df[param1]))))
        plt.title('Chi2 Map Over Params')
        plt.grid(False, color='k')  #, alpha=0.5)
        plt.colorbar()
        plt.gca().xaxis.tick_bottom()

        if save is True:
            out = "{}_param_degens_disk{}.pdf".format(self.out_path, DI)
            plt.savefig(out)
            print(("Saved to ", out))
        else:
            plt.show() #block=False)
        return None


    def plot_all(self):
        self.plot_best_fit_params(save=True)
        self.plot_step_duration(save=True)
        self.DMR_spectra(save=True)
        self.DMR_images(save=True)

# ex_run = GridSearch_Analysis('gridsearch_runs/jan21_hco/jan21_hco')
# ex_run.plot_best_fit_params()





# The End
