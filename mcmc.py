"""Set up and make an MCMC run."""


import os
import sys
import emcee
import pickle
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess as sp
# import matplotlib; matplotlib.use('TkAgg')  # This must be imported first
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter
from disk_model.disk import Disk
from astropy.io import fits

from constants import today, lines #, mol
from tools import already_exists, remove, imstat
#from analysis import plot_fits
#from four_line_run_driver import make_fits
import fitting
# import plotting
import run_driver
# import analysis
import tools

from pathlib2 import Path
Path.cwd()

sns.set_style('ticks')


run_w_pool = True
nwalkers, nsteps = 50, 500


class MCMCrun:
    """A container for an MCMC run and the resulting data.

    By the time this is run, a chain file has already been made.
    It's a little confusing how this integrates with the one-/four-line fits.
    self.mol should be a thing for the one-line fits.
    """
    # Set up a path for the images to go to:
    def __init__(self, run_path, name, mol='hco', nwalkers=nwalkers, burn_in=0):
        """Set up.

        Args:
            run_path (str): the file path to where to find the chain, i.e. './mcmc_runs/feb5/'
            name (str): the name of the run (for output files; usually just '[today]-[run number of the day]').
            nwalkers (int): how many walkers to have.
            burn_in (int): how many of the first steps to ignore.
        """
        self.name            = name
        self.runpath         = run_path + name
        self.image_outpath   = './mcmc_results/' + name
        self.modelfiles_path = run_path + 'model_files/' + name
        self.main            = pd.read_csv(self.runpath + '_chain.csv')
        self.mol             = self.get_line()
        # The 'encoding' arg is to rectify confusion that comes up when Pickle
        # in Py3 tries to read a pickled object that was created in Py2.
        self.param_dict      = pickle.load(open(run_path + 'param_dict.pkl', 'rb'),
                                           encoding='latin1')
        # I think below is equivalent to ^^?
        # self.param_dict      = pickle.load(open(run_path + 'param_dict.pkl', 'r'))

        self.nwalkers = nwalkers
        self.data_path  = './data/{}/{}-short{}'.format(self.mol, self.mol,
                                                        lines[self.mol]['baseline_cutoff'])

        # This only makes sense if it already exists?
        self.nsteps = self.main.shape[0] // nwalkers

        # Remove burn in
        self.burnt_in = self.main.iloc[burn_in*nwalkers:, :]

        # Get rid of steps that resulted in bad lnprobs
        # I think this might be redundant; no -infs since we immediately reject
        # them in the priors check?
        lnprob_vals = self.burnt_in.loc[:, 'lnprob']
        self.groomed = self.burnt_in.loc[lnprob_vals != -np.inf,
                                         :].reset_index().drop('index', axis=1)
        # print 'Removed burn-in phase (step 0 through {}).'.format(burn_in)

        # Get a dictionary of the best-fit param values (for use in structure plotting)
        self.get_bestfit_dict()


        with open(self.runpath + '_log.txt', 'w') as f:
            s0 = 'Run: ' + self.runpath + '\n'
            s1 = 'Molecular line: ' + self.mol + '\n'
            s2 = 'Nwalkers: ' + str(nwalkers) + '\n'
            s3 = 'Nsteps: ' + str(nsteps)  + '\n'
            s = s0 + s1 + s2 + s3
            f.write(s)


    def get_line(self):
        for mol in ['hco', 'hcn', 'co', 'cs']:
            if mol in self.runpath:
                break
        return mol


    def get_bestfit_dict(self):
        subset_df = self.main
        # Locate the best fit model from max'ed lnprob.
        max_lnp = subset_df['lnprob'].max()
        model_params = subset_df[subset_df['lnprob'] == max_lnp].drop_duplicates()

        print('Model parameters:\n') #, [mp, model_params[mp], '\n' for mp in list(model_params)], '\n\n'
        for mp in list(model_params):
            print(mp, round(model_params[mp].values[0], 2))

        # Make a complete dictionary of all the parameters
        self.bf_param_dict = self.param_dict.copy()
        for param in model_params.columns[:-1]:
            self.bf_param_dict[param] = model_params[param].values[0]
        return None


    def get_disk_objects(self):
        self.get_bestfit_dict()
        param_dict = self.bf_param_dict
        print(param_dict)
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
        self.diskA, self.diskB = d1, d2
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


    def corner(self, variables=None, save=True, save_to_thesis=False):
        """Plot 'corner plot' of fit."""
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
            print("Entering else")
            corner_plt.map_lower(sns.kdeplot, cut=0, cmap='Blues',
                                 n_levels=18, shade=True)
            corner_plt.map_lower(sns.kdeplot, cut=0, cmap='Blues',
                                 n_levels=5)

        print("finished conditional")
        # This is where the error is coming from:
        # ValueError: zero-size array to reduction operation minimum which has no identity
        # corner.map_lower(sns.kdeplot, cut=0, cmap='Blues', n_levels=3, shade=False)
        corner_plt.map_diag(sns.kdeplot, cut=0)

        print("Made it this far")
        if variables is None:
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
            table_ax = corner_plt.fig.add_axes([0, 0, 1, 1], frameon=False)
            table_ax.axis('off')
            left, bottom = 0.15, 0.83
            pd.plotting.table(table_ax, stats.round(2), bbox=[
                              left, bottom, 1-left, .12], edges='open', colLoc='right')

            corner_plt.fig.suptitle(r'{} Parameters, {} Walkers, {} Steps $\to$ {} Samples'
                                .format(self.groomed.shape[1], self.nwalkers,
                                        self.groomed.shape[0]//self.nwalkers, self.groomed.shape[0],
                                        fontsize=25))
            tag = ''
        else:
            tag = '_subset'

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



        # bestfit = self.groomed[np.where(np.min(chi)==chi)]
        # sh = np.shape(bestfit)
        # bfs=False
        # if sh!=(ndim,):
        #     nm, nd = sh
        #     for i in range(nm):
        #         for j in range(nm):
        #             if (bestfit[i]!=bestfit[j]).any():
        #                 bfs=True
        #                 break
        #         if bfs:
        #             break
        #     if bfs:
        #         print '########## WARNING: MORE THAN 1 BEST FIT MODEL ##########'
        #     bestfit=bestfit[0].tolist()

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


    def plot_surf_dens(self, save=False):
        """
        Plot density structure of the disk.
        Drawn from Kevin's code.
        """

        # Generate the Disk objects (from Kevin's code)
        # Can be nice (in testing) to manually insert these and not regen them every time.

        param_dict = self.bf_param_dict

        gamma = 1
        R_c = 150 * 1.5e13 # 150 AU in cm
        # m_gas = 0.1 * 2e33 # 0.1 * Solar mass, in grams
        m_gas = 10**self.bf_param_dict['m_disk_A'] * 2e33
        log_r_out = np.log(self.bf_param_dict['r_out_A'] * 1.5e13 )
        # rs = np.linspace(1*1.5e13, 1000*1.5e13, 100)
        rs = np.logspace(13, log_r_out, 100)

        surf_dens = (m_gas * (2 - gamma))/(2*np.pi*R_c**2) * (rs/R_c)**2 * np.exp(-(rs/R_c)**(2-gamma))

        fig, ax = plt.subplots()
        ax.loglog(rs/1.5e13, surf_dens)
        # ax.set_ylim(1e-4, 1e4)
        ax.set_xlim(0, self.bf_param_dict['r_out_A'] * 1.5e13)
        plt.show(block=False)


    def plot_structure(self, zmax=150, save=False, save_to_thesis=False):
        """
        Plot temperature and density structure of the disk.
        Drawn from Kevin's code.
        """

        # Generate the Disk objects (from Kevin's code)
        # Can be nice (in testing) to manually insert these and not regen them every time.
        if not hasattr(self, 'diskA'):
            self.get_disk_objects()
        d1, d2 = self.diskA, self.diskB


        param_dict = self.bf_param_dict
        rmax_a, rmax_b = self.bf_param_dict['r_out_A'], self.bf_param_dict['r_out_B']
        rmax = max(rmax_a, rmax_b)

        # fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True,
        #                                  gridspec_kw = {'width_ratios':[rmax_a, rmax_b]}) # , sharex=True)
        fig, ((cbar_axes), (im_axes)) = plt.subplots(2, 2, figsize=(18, 4), # sharex=True,
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
                                                # np.arange(0, 11, 0.1),
                                                cmap='inferno', levels=50)

            col_dens_contours = im_axes[i].contour(d.r[0,:,:]/d.AU, d.Z[0,:,:]/d.AU,
                                                np.log10(d.sig_col[0,:,:]), (-3, -2,-1),
                                                linestyles=':', linewidths=3,
                                                colors='k')

            temp_contours = im_axes[i].contour(d.r[0,:,:]/Disk.AU, d.Z[0,:,:]/Disk.AU,
                                            d.T[0,:,:], (50, 100, 150), #(20, 40, 60, 80, 100, 120),
                                            colors='b', linestyles='--')

            im_axes[i].clabel(col_dens_contours, fmt='%1i', manual=lab_locs[i])

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
        im_axes[0].set_ylabel('Disk Scale Height (AU)',fontsize=16, weight='bold')
        im_axes[0].set_xlabel('Disk A Outer Radius (AU)',fontsize=16, weight='bold')
        im_axes[1].set_xlabel('Disk B Outer Radius (AU)',fontsize=16, weight='bold')

        im_axes[0].set_xlim(rmax_a, 0)
        im_axes[0].set_ylim(-zmax, zmax)

        im_axes[1].set_xlim(0, rmax_b)
        im_axes[1].set_ylim(-zmax, zmax)

        im_axes[0].yaxis.set_ticks_position('left')
        im_axes[1].yaxis.set_ticks_position('right')


        # cbar_axes[0].set_title('Disk A', weight='bold')
        # cbar_axes[1].set_title('Disk B', weight='bold')
        cbar_axes[0].xaxis.set_ticks_position('top')
        cbar_axes[1].xaxis.set_ticks_position('top')

        cbar_axes[1].xaxis.set_label_position('top')
        cbar_axes[1].xaxis.set_label_position('top')


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


    def DMR_images(self, cmap='RdBu', save=False, save_to_thesis=False):
        """
        Plot a triptych of data, model, and residuals.

        It would be nice to have an option to also plot the grid search results.
        Still need to:
        - Get the beam to plot
        - Get the velocity labels in the right places

        Some nice cmaps: magma, rainbow
        """
        plt.close()
        print("\nPlotting DMR images...")


        data_path  = self.data_path + '.fits'
        resid_path = self.modelfiles_path + '_bestFit_resid.fits'
        model_path = self.modelfiles_path + '_bestFit.fits'
        if not Path(model_path).exists():
            print("No best-fit model made yet; making now...")
            self.make_best_fits(plot_bf=False)

        real_data    = fits.getdata(data_path, ext=0).squeeze()
        model_data   = fits.getdata(model_path, ext=0).squeeze()
        resid_data   = fits.getdata(resid_path, ext=0).squeeze()
        image_header = fits.getheader(data_path, ext=0)
        model_header = fits.getheader(model_path, ext=0)
        resid_header = fits.getheader(resid_path, ext=0)

        # Define some plotting params
        hspace = -0.
        wspace = -0.

        # Set up some physical params
        # vmin, vmax = np.nanmin(real_data), np.nanmax(real_data)
        vmax = np.nanmax((np.nanmax(real_data), -np.nanmin(real_data)))
        vmin = -vmax

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
        nchans = 30

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
        n_rows = 3
        n_cols = int(np.ceil(nchans/3))

        # Get the plots going
        # fig = plt.figure(figsize=(n_rows * 3, 7))
        fig = plt.figure(figsize=(18, 18))
        big_fig = gridspec.GridSpec(3, 1)

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
            ax_d = plt.Subplot(fig, data_ims[i])
            ax_m = plt.Subplot(fig, model_ims[i])
            ax_r = plt.Subplot(fig, resid_ims[i])

            if i == int(np.floor(n_cols / 2)):
                ax_d.set_title('Data', weight='bold')
                ax_m.set_title('Model', weight='bold')
                ax_r.set_title('Residuals', weight='bold')

            # Plot the data
            im_d = ax_d.contourf(real_data[i + chan_offset][xmin:xmax, xmin:xmax],
                                 levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
            im_m = ax_m.contourf(model_data[i + chan_offset][xmin:xmax, xmin:xmax],
                                 levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
            im_r = ax_r.contourf(resid_data[i + chan_offset][xmin:xmax, xmin:xmax],
                                 levels=30, cmap=cmap, vmin=vmin, vmax=vmax)


            # Add n-sigma contours
            im_d = ax_d.contour(real_data[i + chan_offset][xmin:xmax, xmin:xmax],
                                 levels=contours)
            im_m = ax_m.contour(model_data[i + chan_offset][xmin:xmax, xmin:xmax],
                                 levels=contours)
            im_r = ax_r.contour(resid_data[i + chan_offset][xmin:xmax, xmin:xmax],
                                 levels=contours)




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
            ax_d.text(44, 80, velocity + ' km/s', fontsize=6, color='k',
                    horizontalalignment='center', verticalalignment='center')
            ax_m.text(44, 80, velocity + ' km/s', fontsize=6, color='k',
                    horizontalalignment='center', verticalalignment='center')
            ax_r.text(44, 80, velocity + ' km/s', fontsize=6, color='k',
                    horizontalalignment='center', verticalalignment='center')

            if i == n_cols * (n_rows - 1) and add_beam_d is True:
                el = Ellipse(xy=[0.8 * crop_arcsec, 0.8 * crop_pix],
                             width=bmin, height=bmaj, angle=-bpa,
                             fc='k', ec='k', fill=False, hatch='////////')
                ax_d.add_artist(el)

            fig.add_subplot(ax_m)
            fig.add_subplot(ax_d)
            fig.add_subplot(ax_r)
            fig.tight_layout()



        fig.tight_layout()
        fig.subplots_adjust(wspace=0., hspace=0.1, top=0.95, right=0.9)

        cmaps = plt.imshow(real_data[i + chan_offset][xmin:xmax, xmin:xmax],
                           extent=(crop_arcsec, -crop_arcsec, crop_arcsec, -crop_arcsec),
                           cmap=cmap, vmin=vmin, vmax=vmax)

        cax = plt.axes([0.91, 0.01, 0.03, 0.94]) # [l, b, w, h]
        cbar = plt.colorbar(cmaps, cax=cax, orientation='vertical')
        cbar.set_label('Jy/beam', labelpad=-30, fontsize=20, weight='bold')
        cbar.set_ticks(np.linspace(vmin, vmax, 6))



        out_path = self.image_outpath + '_DMR-images.png'
        thesis_fig_path = '../Thesis/Figures/chanmaps-{}.pdf'.format(self.mol)
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
        # pv_ticks = np.linspace(min(pv_ts), max(pv_ts), 5) - np.mean(pv_ts)

        start, end = ax_pv.get_xlim()
        pv_tick_labels = (np.linspace(start, end, 5) - np.mean([start, end])) * pixel_to_AU
        pv_tick_labels = [int(tick) for tick in pv_tick_labels]

        vmin, vmax = ax_pv.get_ylim()
        vel_tick_labels = np.linspace(vmin, vmax, 5) - np.mean([vmin, vmax])
        vel_tick_labels = [int(tick) for tick in vel_tick_labels]


        # ax_pv.set_xticklabels(pv_tick_labels)
        # ax_pv.set_yticklabels(vel_tick_labels)
        # ax_pv.set_ylabel("Velocity (km/s)", weight='bold', rotation=270)
        # ax_pv.set_xlabel("Position Offset (AU)", weight='bold')
        # ax_pv.yaxis.tick_right()
        # ax_pv.yaxis.set_label_position("right")
        #
        #
        # start, end = ax_image.get_xlim()
        # image_xtick_labels = (np.linspace(start, end, 5) - np.mean([start, end])) * pixel_to_AU
        # image_xtick_labels = [int(tick) for tick in image_xtick_labels]
        #
        # start, end = ax_image.get_ylim()
        # image_ytick_labels = (np.linspace(start, end, 5) - np.mean([start, end])) * pixel_to_AU
        # image_ytick_labels = [int(tick) for tick in image_ytick_labels]
        #
        #
        # x_ts = np.array(ax_image.get_xticks().tolist()) * pixel_to_AU
        # # image_xticks = np.linspace(min(x_ts), max(x_ts), 5) - np.mean(x_ts)
        # # image_xtick_labels = [int(tick) for tick in image_xticks]
        # #
        # y_ts = np.array(ax_image.get_yticks().tolist()) * pixel_to_AU
        # # image_yticks = np.linspace(min(y_ts), max(y_ts), 5) - np.mean(y_ts)
        # # image_ytick_labels = [int(tick) for tick in image_yticks]
        #
        # # ax_image.set_xticklabels(x_ts)
        # # ax_image.set_yticklabels(y_ts)
        # ax_image.set_xticklabels(image_xtick_labels)
        # ax_image.set_yticklabels(image_ytick_labels)
        # ax_image.set_xlabel("Position Offset (AU)", weight='bold')
        # ax_image.set_ylabel("Position Offset (AU)", weight='bold')

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









def run_emcee(run_path, run_name, mol, nsteps, nwalkers, lnprob, pool):
    """
    Make an actual MCMC run.

    Other than in setting up param_info, this is actually line-agnostic.
    The line-specificity is created in the lnprob function.

    Args:
        run_path (str): the name to output I guess
        run_name (str): the name to feed the actual emcee routine (line 360)
        mol (str): which line we're running.
        nsteps (int): How many steps we're taking.
        nwalkers (int): How many walkers we're using
        lnprob (func): The lnprob function to feed emcee
        param_info (list): list of [param name,
                                    initial_position_center,
                                    initial_position_sigma,
                                    (prior low bound, prior high bound)]
                            for each parameter.
                            The second two values set the position & size
                            for a random Gaussian ball of initial positions
    """
    # Name the chain we're looking for
    chain_filename = run_path + run_name + '_chain.csv'


    # Note that this is what is fed to MCMC to dictate how the walkers move, not
    # the actual set of vars that make_fits pulls from.
    # ORDER MATTERS here (for comparing in lnprob)
    # Values that are commented out default to the starting positions in run_driver/param_dict
    # Note that param_info is of form:
    # [param name, init_pos_center, init_pos_sigma, (prior lower, prior upper)]

    # ALL BUT CO
    if mol != 'co':
        param_info = [('r_out_A',           500,     300,      (10, 700)),
                      ('atms_temp_A',       300,     150,      (0, 1000)),
                      ('mol_abundance_A',   -8,      3,        (-13, -3)),
                      ('temp_struct_A',     -0.,      1.,      (-3., 3.)),
                      # ('incl_A',            65.,     30.,      (0, 90.)),
                      ('pos_angle_A',       70,      45,       (0, 360)),
                      ('r_out_B',           500,     300,      (10, 400)),
                      ('atms_temp_B',       200,     150,      (0, 1000)),
                      ('mol_abundance_B',   -8,      3,        (-13, -3)),
                      # ('temp_struct_B',     0.,      1,        (-3., 3.)),
                      # ('incl_B',            45.,     30,       (0, 90.)),
                      ('pos_angle_B',       136.0,   45,       (0, 180))
                      ]
    # CO ONLY
    else:
        param_info = [('r_out_A',           500,     300,      (10, 700)),
                      ('atms_temp_A',       300,     150,      (0, 1000)),
                      ('m_disk_A',          -1.,      1.,      (-4.5, 0)),
                      ('temp_struct_A',     -0.,      1.,      (-3., 3.)),
                      # ('incl_A',            65.,     30.,      (0, 90.)),
                      ('pos_angle_A',        70,      45,      (0, 180)),
                      ('r_out_B',           500,     300,      (10, 400)),
                      ('atms_temp_B',       200,     150,      (0, 1000)),
                      ('m_disk_B',          -4.,      1.,      (-6., 0)),
                      # ('temp_struct_B',     0.,      1,        (-3., 3.)),
                      # ('incl_B',            45.,     30,       (0, 90.)),
                      # ('pos_angle_B',       136.0,   45,       (0, 180))
                      ]


    print("Setting up directories for new run")
    remove(run_path)
    sp.call(['mkdir', run_path])
    sp.call(['mkdir', run_path + '/model_files'])

    # Export the initial param dict for accessing when we want
    pickle.dump(run_driver.param_dict, open(run_path + 'param_dict.pkl', 'wb'))
    print("Wrote {}param_dict.pkl out".format(run_path))
    print('Starting {}'.format(run_path))

    start_step = 0

    # Start a new file for the chain; set up a header line
    with open(chain_filename, 'w') as f:
        param_names = [param[0] for param in param_info]
        np.savetxt(f, (np.append(param_names, 'lnprob'), ),
                   delimiter=',', fmt='%s')

    # Set up initial positions
    # randn makes an n-dimensional array of rands in [0,1]
    # param[2] is the sigma of the param
    # This is bad because it doesn't check for positions outside of priors.
    # pos = [[param[1] + param[2]*np.random.randn()
    #         for param in param_info]
    #        for i in range(nwalkers)]

    pos = []
    for i in range(nwalkers):
        pos_walker = []
        for param in param_info:
            pos_i = param[1] + param[2]*np.random.randn()
            # Make sure we're starting within priors
            lower_bound, upper_bound = param[-1]
            # If it is, put it into the dict that make_fits calls from
            # while pos_i < param[3][0] or pos_i > param[3][1]:
            while not lower_bound < pos_i < upper_bound:
                pos_i = param[1] + param[2]*np.random.randn()

            pos_walker.append(pos_i)
        pos.append(pos_walker)

    # Initialize sampler chain
    # Recall that param_info is a list of length len(d1_params)+len(d2_params)
    # There's gotta be a more elegant way of doing this.


    print("Initializing sampler.")
    ndim = len(param_info)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(run_name, param_info, mol),
                                    pool=pool)

    # Initiate a generator to provide the data. They changed the arg
    # storechain -> store sometime between v2.2.1 (iorek) and v3.0rc2 (cluster)
    from emcee import __version__ as emcee_version
    print("About to run sampler")
    if emcee_version[0] == '2':     # Linux boxes are on v2, cluster is v3
        run = sampler.sample(pos, iterations=nsteps, storechain=False)  # for iorek
    else:
        run = sampler.sample(pos, iterations=nsteps, store=False)   # for cluster
    """Note that sampler.sample returns:
            pos: list of the walkers' current positions in an object of shape
                    [nwalkers, ndim]
            lnprob: The list of log posterior probabilities for the walkers at
                    positions given by pos.
                    The shape of this object is (nwalkers, dim)
            rstate: The current state of the random number generator.
            blobs (optional): The metadata "blobs" associated with the current
                              position. The value is only returned if
                              lnpostfn returns blobs too.
            """
    lnprobs = []
    # import pdb; pdb.set_trace()


    # return run
    print("THIS CURRENT WORKING DIRECTORY IS" + os.getcwd() + '\n\n')
    print("About to loop over run")
    for i, result in enumerate(run):
        print("Got a result")

        # Maybe do this logging out in the lnprob function itself?
        pos, lnprobs, blob = result
        # print "Lnprobs: ", lnprobs

        # Log out the new positions
        with open(chain_filename, 'a') as f:
            new_step = [np.append(pos[k], lnprobs[k]) for k in range(nwalkers)]
            print("Adding a new step to the chain: ", new_step)
            np.savetxt(f, new_step, delimiter=',')

    print("Ended run")









# The End


# The End
