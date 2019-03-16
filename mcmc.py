"""Set up and make an MCMC run."""


import sys
import emcee
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from emcee.utils import MPIPool
from constants import today #, mol
from tools import already_exists, remove
#from analysis import plot_fits
#from four_line_run_driver import make_fits
import fitting
# import plotting
import run_driver
import analysis
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
        self.mol             = mol
        self.runpath         = run_path + name
        self.image_outpath   = './mcmc_results/' + name
        self.modelfiles_path = run_path + 'model_files/' + name
        self.main            = pd.read_csv(self.runpath + '_chain.csv')
        self.param_dict      = pickle.load(open(run_path + 'param_dict.pkl', 'rb'))

        """
        if path:
            self.main = pd.read_csv(path + '.csv')
        else:
            self.main = pd.read_csv(name + '/' + name + '_chain.csv')
        """

        self.nwalkers = nwalkers


        # This only makes sense if it already exists?
        self.nsteps = self.main.shape[0] // nwalkers

        # Remove burn in
        self.burnt_in = self.main.iloc[burn_in*nwalkers:, :]

        # Get rid of steps that resulted in bad lnprobs
        lnprob_vals = self.burnt_in.loc[:, 'lnprob']
        self.groomed = self.burnt_in.loc[lnprob_vals != -np.inf, :]
        # print 'Removed burn-in phase (step 0 through {}).'.format(burn_in)



        with open(self.runpath + '_log.txt', 'w') as f:
            s0 = 'Run: ' + self.runpath + '\n'
            s1 = 'Molecular line: ' + mol + '\n'
            s2 = 'Nwalkers: ' + str(nwalkers) + '\n'
            s3 = 'Nsteps: ' + str(nsteps)  + '\n'
            s = s0 + s1 + s2 + s3
            f.write(s)


    def evolution(self, save=True):
        """Plot walker evolution.

        Uses groomed data, so no infs.
        """
        print 'Making walker evolution plot...'
        plt.close()

        self.nsteps = len(self.groomed)//self.nwalkers
        stepmin, stepmax = 0, self.nsteps
        print self.nsteps, self.nwalkers

        main = self.groomed.copy().iloc[stepmin * self.nwalkers:
                                     stepmax * self.nwalkers, :]

        axes = main.iloc[0::self.nwalkers].plot(
            x=np.arange(stepmin, stepmax),
            figsize=(9, 1.5*(len(main.columns))),
            subplots=True,
            color='black',
            alpha=0.1)

        for i in range(self.nwalkers-1):
            main.iloc[i+1::self.nwalkers].plot(
                x=np.arange(stepmin, stepmax), subplots=True, ax=axes,
                legend=False, color='black', alpha=0.1)

            # make y-limits on lnprob subplot reasonable
            amin = main.iloc[-1 * self.nwalkers:, -1].min()
            amax = main.lnprob.max()
            axes[-1].set_ylim(amin, amax)
            #axes[-1].set_ylim(-40000, 0)

        # if you want mean at each step over plotted:
        # main.index //= self.nwalkers
        # walker_means = pd.DataFrame([main.loc[i].mean() for i in range(self.nsteps)])
        # walker_means.plot(subplots=True, ax=axes, legend=False, color='forestgreen', ls='--')

        plt.tight_layout()
        plt.suptitle(self.name + ' walker evolution')
        if save:
            plt.savefig(self.image_outpath + '_evolution.pdf')
            print 'Image saved image to ' + self.image_outpath + '_evolution.pdf'
        else:
            plt.show()


    def evolution_main(self, min_lnprob=None, save=True):
        """Plot walker evolution.

        This one uses the full step log, including bad steps (not groomed).
        """
        print 'Making walker evolution plot...'
        plt.close()

        self.nsteps = len(self.main)//self.nwalkers
        stepmin, stepmax = 0, self.nsteps
        print self.nsteps, self.nwalkers

        main = self.main.copy().iloc[stepmin * self.nwalkers:
                                     stepmax * self.nwalkers, :]

        axes = main.iloc[0::self.nwalkers].plot(
            x=np.arange(stepmin, stepmax),
            figsize=(9, 1.5*(len(main.columns))),
            subplots=True,
            color='black',
            alpha=0.1)

        for i in range(self.nwalkers-1):
            main.iloc[i+1::self.nwalkers].plot(
                x=np.arange(stepmin, stepmax), subplots=True, ax=axes,
                legend=False, color='black', alpha=0.1)

            # make y-limits on lnprob subplot reasonable
            # This is not reasonable. Change it?
            #axes[-1].set_ylim(main.iloc[-1 * self.nwalkers:, -1].min(), main.lnprob.max())
            amin = np.amin(main.lnprob[main.lnprob != -np.inf]) if not min_lnprob else min_lnprob
            amax = np.amax(main.lnprob)
            axes[-1].set_ylim(amin, amax)
            # axes[-1].set_ylim(-50000, -25000)


        # A quick iPython walker lnprob plotter:
        # for i in range(nwalkers):
        #   plt.plot(run.main['lnprob'][i+1::nwalkers], linewidth=0.2)

        # if you want mean at each step over plotted:
        # main.index //= self.nwalkers
        # walker_means = pd.DataFrame([main.loc[i].mean() for i in range(self.nsteps)])
        # walker_means.plot(subplots=True, ax=axes, legend=False, color='forestgreen', ls='--')

        plt.tight_layout()
        plt.suptitle(self.name + ' walker evolution')

        if save:
            plt.savefig(self.image_outpath + '_evolution-main.pdf')  # , dpi=1)
            print 'Image saved image to ' + self.image_outpath + '_evolution-main.pdf'
        else:
            plt.show()


    def kde(self):
        """Make a kernel density estimate (KDE) plot."""
        print 'Generating posterior kde plots...'
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
        print 'Image saved image to ' + self.image_outpath + '_kde.png'
        plt.show()


    def corner(self, variables=None, save=True):
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
        print(stats.T.round(6).to_string())

        # make corner plot
        print "Starting SNS PairGrid"
        corner_plt = sns.PairGrid(data=self.groomed, diag_sharey=False, despine=False, vars=variables)

        print "Finished SNS PairGrid"

        if variables is not None:
            print "Entering if"
            corner_plt.map_lower(plt.scatter, s=1, color='#708090', alpha=0.1)
        else:
            print "Entering else"
            corner_plt.map_lower(sns.kdeplot, cut=0, cmap='Blues',
                                 n_levels=18, shade=True)
            corner_plt.map_lower(sns.kdeplot, cut=0, cmap='Blues',
                                 n_levels=5)

        print "finished conditional"
        # This is where the error is coming from:
        # ValueError: zero-size array to reduction operation minimum which has no identity
        # corner.map_lower(sns.kdeplot, cut=0, cmap='Blues', n_levels=3, shade=False)
        corner_plt.map_diag(sns.kdeplot, cut=0)

        print "Made it this far"
        if variables is None:
            # get best_fit and posterior statistics
            stats = self.groomed.describe(percentiles=[0.16, 0.84]).drop([
                'count', 'min', 'max', 'mean'])
            stats.loc['best fit'] = self.main.loc[self.main['lnprob'].idxmax()]
            stats = stats.iloc[[-1]].append(stats.iloc[:-1])
            stats.loc[['16%', '84%'], :] -= stats.loc['50%', :]
            stats = stats.reindex(
                ['50%', '16%', '84%', 'best fit', 'std'], copy=False)
            print(stats.T.round(3).to_string())
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
            plt.savefig(self.image_outpath + '_corner.pdf')
            print 'Image saved image to ' + self.image_outpath + '_corner.png'
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
        print(stats.T.round(6).to_string())



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
        for df, disk_ID in zip([groomed_diskA, groomed_diskB], ('A', 'B')):
            print "Making corner plot"
            corner.corner(df, quantiles=[0.16,0.5,0.84], verbose=False, show_titles=True, truths=bestfit)#, labels=labels,title_args={'fontsize': 12})

            plt.savefig('{}cornerplot-{}-disk{}.png'.format(outpath, mol, disk_ID), dpi=200)
            print "Saved plot for disk{} to '{}cornerplot-{}-disk{}.png'.format(outpath, mol, disk_ID)'"

        # corner.corner(groomed_diskA, quantiles=[0.16,0.5,0.84], verbose=False, show_titles=True, truths=bestfit)#, labels=labels,title_args={'fontsize': 12})

        # corner.corner(groomed_diskB, quantiles=[0.16,0.5,0.84], verbose=False, show_titles=True, truths=bestfit)#, labels=labels,title_args={'fontsize': 12})


        # plt.savefig(self.image_outpath + '_corner.pdf')
        # print "Saved plot."

    def make_best_fits(self):
        """Do some modeling stuff.

        Args:
            run (mcmc.MCMCrun): the
        """
        # run.main is the pd.df that gets read in from the chain.csv
        subset_df = self.main  # [run.main['r_in'] < 15]

        # Locate the best fit model from max'ed lnprob.
        max_lnp = subset_df['lnprob'].max()
        model_params = subset_df[subset_df['lnprob'] == max_lnp].drop_duplicates()

        print 'Model parameters:\n' #, [mp, model_params[mp], '\n' for mp in list(model_params)], '\n\n'
        for mp in list(model_params):
            print mp, model_params[mp].values[0]

        # Check if we're looking at a one- or four-line fit.
        fourlinefit_tf = True if 'r_out_A-cs' in model_params.columns else False

        # Make a complete dictionary of all the parameters
        bf_param_dict = self.param_dict.copy()
        for param in model_params.columns[:-1]:
            bf_param_dict[param] = model_params[param].values

        bf_disk_params = bf_param_dict.keys()

        print 'Making model...'

        def make_model(param_dict, mol, fourlinefit_tf=False):
            # If we're doing four-line fitting, then some values are dictionaries of vals.
            # Just want the relevant line.
            param_dict_mol = param_dict.copy()
            for p in param_dict_mol:
                if type(param_dict_mol[p]) == dict:
                    param_dict_mol[p] = param_dict_mol[p][mol]

            obs = fitting.Observation(mol, cut_baselines=True)
            model = fitting.Model(observation=obs,
                                  run_name=self.name,
                                  model_name=self.name + '_bestFit')
            run_driver.make_fits(model, param_dict_mol, mol)
            tools.sample_model_in_uvplane(model.modelfiles_path, mol=mol)
            tools.icr(model.modelfiles_path, mol=mol)
            tools.plot_fits(model.modelfiles_path + '.fits', mol=mol,
                               best_fit=True, save=True)
            return model

        models = []
        # If it's a one line fit, it's easy.
        if not fourlinefit_tf:
            models.append(make_model(bf_param_dict, bf_param_dict['mol']))

        else:
            # This assumes that outer radius and abundance are the only things
            # being individually varied. Maybe coordinate better.
            for m in ['cs', 'co', 'hco', 'hcn']:
                # Doesn't matter that we've still got the other lines' info here
                bf_param_dict['r_out_A'] = float(bf_param_dict['r_out_A-{}'.format(m)])
                bf_param_dict['r_out_B'] = float(bf_param_dict['r_out_B-{}'.format(m)])
                bf_param_dict['mol_abundance_A'] = float(bf_param_dict['mol_abundance_A-{}'.format(m)])
                bf_param_dict['mol_abundance_B'] = float(bf_param_dict['mol_abundance_B-{}'.format(m)])

                models.append(make_model(bf_param_dict, m))




        # This seems cool but I need to get plotting.py going first.
        #"""
        """
        fig = plotting.Figure(layout=(1, 3),
                              paths=[aumic_fitting.band6_fits_images[-1],
                                     paths[-1] + '.fits',
                                     paths[-1] + '.residuals.fits'],
                              rmses=3*[aumic_fitting.band6_rms_values[-1]],
                              texts=[[[4.6, 4.0, 'Data']],
                                     [[4.6, 4.0, 'Model']],
                                     [[4.6, 4.0, 'Residuals']]
                                     ],
                              title=r'Run 6 Global Best Fit Model & Residuals',
                              savefile=run.name + '/' + run.name + '_bestfit_concise.pdf',
                              show=True)
        """
        # plot_fits(self.runpath)

        return (models, bf_param_dict)




def run_emcee(run_path, run_name, mol, nsteps, nwalkers, lnprob):
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
    # print "Made it to run_emcee"
    # Name the chain we're looking for
    chain_filename = run_path + run_name + '_chain.csv'

    # Set up the parallelization
    # pool = MPIPool()
    if run_w_pool:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)



    # Note that this is what is fed to MCMC to dictate how the walkers move, not
    # the actual set of vars that make_fits pulls from.
    # ORDER MATTERS here (for comparing in lnprob)
    # Note that param_info is of form:
    # [param name, init_pos_center, init_pos_sigma, (prior lower, prior upper)]
    if mol != 'co':
        param_info = [('r_out_A',           500,     300,      (10, 1000)),
                      ('atms_temp_A',       300,     150,      (0, 1000)),
                      ('mol_abundance_A',   -8,      3,        (-13, -3)),
                      ('temp_struct_A',    -0.,      1.,       (-3., 3.)),
                      ('incl_A',            65.,     30.,      (0, 90.)),
                      ('pos_angle_A',       70,      45,       (0, 360)),
                      ('r_out_B',           500,     300,      (10, 1000)),
                      ('atms_temp_B',       200,     150,      (0, 1000)),
                      ('mol_abundance_B',   -8,      3,        (-13, -3)),
                      # ('temp_struct_B',     0.,      1,        (-3., 3.)),
                      ('incl_B',            45.,     30,       (0, 90.)),
                      ('pos_angle_B',       136.0,   45,       (0, 360))
                      ]

    else:
        param_info = [('r_out_A',           500,     300,      (10, 1000)),
                      ('atms_temp_A',       300,     150,      (0, 1000)),
                      ('m_disk_A',          -1.,      1.,      (-4.5, 0)),
                      ('temp_struct_A',     -0.,      1.,      (-3., 3.)),
                      ('incl_A',            65.,     30.,      (0, 90.)),
                      ('pos_angle_A',        70,      45,      (0, 360)),
                      ('r_out_B',           500,     300,      (10, 1000)),
                      ('atms_temp_B',       200,     150,      (0, 1000)),
                      ('m_disk_B',          -4.,      1.,      (-6., 0)),
                      # ('temp_struct_B',     0.,      1,        (-3., 3.)),
                      ('incl_B',            45.,     30,       (0, 90.)),
                      ('pos_angle_B',       136.0,   45,       (0, 360))
                      ]

    # Try to resume an existing run of this name.
    # There's gotta be a more elegant way of doing this.
    if already_exists(chain_filename) is True:
        resume = True if len(pd.read_csv(chain_filename).index) > 0 else False
    else:
        resume = False

    if resume is True:
        print "Resuming run"
        chain = pd.read_csv(chain_filename)
        start_step = chain.index[-1] // nwalkers
        pos = np.array(chain.iloc[-nwalkers:, :-1])
        print 'Resuming {} at step {}'.format(run_name, start_step)

        # If we're adding new steps, just put in a new line and get started.
        with open(chain_filename, 'a') as f:
            f.write('\n')
        # Not sure what end looks like.
        end = np.array(chain.iloc[-nwalkers:, :])
        print 'Start step: {}'.format(np.mean(end[:, -1]))

    # If there's no pre-existing run, set one up, and delete any empty dirs.
    else:
        print "Setting up directories for new run"
        remove(run_path)
        sp.call(['mkdir', run_path])
        sp.call(['mkdir', run_path + '/model_files'])

        # Export the initial param dict for accessing when we want
        pickle.dump(run_driver.param_dict, open(run_path + 'param_dict.pkl', 'w'))
        print "Wrote {}param_dict.pkl out".format(run_path)
        print 'Starting {}'.format(run_path)

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
    print "Initializing sampler."
    ndim = len(param_info)
    if run_w_pool is True:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                         args=(run_name, param_info, mol),
                                        pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        args=(run_name, param_info, mol))

    # Initiate a generator to provide the data. More about generators here:
    # https://medium.freecodecamp.org/how-and-why-you-should-use-python-generators-f6fb56650888
    print "About to run sampler"
    run = sampler.sample(pos, iterations=nsteps, storechain=False)
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
    print "About to loop over run"
    for i, result in enumerate(run):
        print "Got a result"

        # Maybe do this logging out in the lnprob function itself?
        pos, lnprobs, blob = result
        # print "Lnprobs: ", lnprobs

        # Log out the new positions
        with open(chain_filename, 'a') as f:
            new_step = [np.append(pos[k], lnprobs[k]) for k in range(nwalkers)]
            print "Adding a new step to the chain: ", new_step
            np.savetxt(f, new_step, delimiter=',')

    print "Ended run"
    if run_w_pool is True:
        pool.close()









# The End
