"""Set up and make an MCMC run."""


import sys
import emcee
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from emcee.utils import MPIPool
#from run_driver import run_path #, run_name
from constants import today, mol #, nwalkers, nsteps
from tools import already_exists, remove
from analysis import plot_fits, plot_model_and_data
import fitting
import plotting
import run_driver



sns.set_style('ticks')

run_w_pool = True
nwalkers, nsteps = 50, 500


class MCMCrun:
    """A container for an MCMC run and the resulting data.

    By the time this is run, a chain file has already been made.
    """
    # Set up a path for the images to go to:
    def __init__(self, run_path, name, nwalkers=nwalkers, burn_in=0):
        """Set up.

        Args:
            path (str): the file path to where to find the chain
            name (str): the name of the run (for output files).
            nwalkers (int): how many walkers to have.
            burn_in (int): how many of the first steps to ignore.
        """
        self.name = name
        self.runpath = run_path + name
        self.image_outpath = './mcmc_results/' + name

        # having name/name_chain.csv makes sense
        self.main = pd.read_csv(self.runpath + '_chain.csv')
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
        print 'Removed burn-in phase (step 0 through {}).'.format(burn_in)





        with open(self.runpath + '_log.txt', 'w') as f:
            s0 = 'Run: ' + self.runpath + '\n'
            s1 = 'Molecular line: ' + mol + '\n'
            s2 = 'Nwalkers: ' + str(nwalkers) + '\n'
            s3 = 'Nsteps: ' + str(nsteps)  + '\n'
            s = s0 + s1 + s2 + s3
            f.write(s)


    def evolution(self):
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
        plt.savefig(self.image_outpath + '_evolution.pdf')  # , dpi=1)
        print 'Image saved image to ' + self.image_outpath + '_evolution.pdf'
        plt.show(block=False)

    def evolution_main(self):
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
            amin = np.nanmin(main.lnprob[main.lnprob != -np.inf])
            amax = np.nanmax(main.lnprob)
            axes[-1].set_ylim(amin, amax)
            #axes[-1].set_ylim(-50000, -38000)

        # if you want mean at each step over plotted:
        # main.index //= self.nwalkers
        # walker_means = pd.DataFrame([main.loc[i].mean() for i in range(self.nsteps)])
        # walker_means.plot(subplots=True, ax=axes, legend=False, color='forestgreen', ls='--')

        plt.tight_layout()
        plt.suptitle(self.name + ' walker evolution')
        plt.savefig(self.image_outpath + '_evolution-main.pdf')  # , dpi=1)
        print 'Image saved image to ' + self.image_outpath + '_evolution-main.pdf'
        plt.show(block=False)

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

    def corner(self, variables=None):
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
        corner = sns.PairGrid(data=self.groomed, diag_sharey=False, despine=False,
                              vars=variables)
        print "Finished SNS PairGrid"

        if variables is not None:
            print "Entering if"
            corner.map_lower(plt.scatter, s=1, color='#708090', alpha=0.1)
        else:
            print "Entering else"
            corner.map_lower(sns.kdeplot, cut=0, cmap='Blues',
                             n_levels=18, shade=True)

        print "finished conditional"
        # This is where the error is coming from:
        # ValueError: zero-size array to reduction operation minimum which has no identity
        # corner.map_lower(sns.kdeplot, cut=0, cmap='Blues', n_levels=3, shade=False)
        corner.map_diag(sns.kdeplot, cut=0)

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
            table_ax = corner.fig.add_axes([0, 0, 1, 1], frameon=False)
            table_ax.axis('off')
            left, bottom = 0.15, 0.83
            pd.plotting.table(table_ax, stats.round(2), bbox=[
                              left, bottom, 1-left, .12], edges='open', colLoc='right')

            corner.fig.suptitle(r'{} Parameters, {} Walkers, {} Steps $\to$ {} Samples'
                                .format(self.groomed.shape[1], self.nwalkers,
                                        self.groomed.shape[0]//self.nwalkers, self.groomed.shape[0],
                                        fontsize=25))
            tag = ''
        else:
            tag = '_subset'

        # hide upper triangle, so that it's a conventional corner plot
        for i, j in zip(*np.triu_indices_from(corner.axes, 1)):
            corner.axes[i, j].set_visible(False)

        # fix decimal representation
        for ax in corner.axes.flat:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))

        plt.subplots_adjust(top=0.9)
        plt.savefig(self.image_outpath + '_corner.png', dpi=200)
        print 'Image saved image to ' + self.image_outpath + '_corner.png'

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
        print 'Model parameters:\n', model_params.to_string(), '\n\n'

        bf_param_dict = {}
        for param in model_params.columns[:-1]:
            bf_param_dict[param] = model_params[param].values

        print bf_param_dict
        bf_disk_params = bf_param_dict.keys()
	return bf_disk_params

        # intialize model and make fits image
        print 'Making model...'
        # This obviously has to be generalized.


        obs = fitting.Observation(mol, cut_baselines=True)
        model = fitting.Model(observation=obs,
                              run_name=self.runpath,
                              model_name=self.name + '_bestFit')
        run_driver.make_fits(model, bf_param_dict)
        analysis.plot_fits(self.runpath + '_bestFit.fits', mol=mol,
                           bestFit=True,)

        # This seems cool but I need to get plotting.py going first.
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

        return (model, bf_param_dict)







# Use this one.
def run_emcee(run_path, run_name, nsteps, nwalkers, lnprob, param_info):
    """Make an actual MCMC run.

    Args:
        run_path (str): the name to output I guess
        run_name (str): the name to feed the actual emcee routine (line 360)
        nsteps (int):
        nwalkers (int):
        lnprob (something):
        param_info (list): list of [param name,
                                    initial_position_center,
                                    initial_position_sigma,
                                    (prior low bound, prior high bound)]
                            for each parameter.
                            The second two values set the position & size
                            for a random Gaussian ball of initial positions
    """
    # Name the chain we're looking for
    chain_filename = run_path + today + '_chain.csv'

    # Set up the parallelization
    pool = MPIPool()
    if run_w_pool:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    # Try to resume an existing run of this name.
    # There's gotta be a more elegant way of doing this.
    if already_exists(chain_filename) is True:
        resume = True if len(pd.read_csv(chain_filename).index) > 0 else False
    else:
        resume = False

    if resume is True:
        chain = pd.read_csv(chain_filename)
        start_step = chain.index[-1] // nwalkers
        pos = np.array(chain.iloc[-nwalkers:, :-1])
        print 'Resuming {} at step {}'.format(run_name, start_step)

        # If we're adding new steps, just put in a new line and get started.
        with open(chain_filename, 'a') as f:
            f.write('\n')
        # Not sure whatend looks like.
        end = np.array(chain.iloc[-nwalkers:, :])
        print 'Start step: {}'.format(np.mean(end[:, -1]))

    # If there's no pre-existing run, set one up, and delete any empty dirs.
    else:
        remove(run_path)
        sp.call(['mkdir', run_path])
        sp.call(['mkdir', run_path + '/model_files'])
        print 'Starting {}'.format(run_path)

        start_step = 0
        # Start a new file for the chain
        # Set up a header line
        with open(chain_filename, 'w') as f:
            param_names = [param[0] for param in param_info]
            np.savetxt(f, (np.append(param_names, 'lnprob'), ),
                       delimiter=',', fmt='%s')

        # Set up initial positions?
        """I think this is saying the same thing as the nested list comps.
        pos = []
        for i in range(nwalkers):
            for param in param_info:
                pos.append(param[1] + param[2]*np.random.randn())
        """
        # randn makes an n-dimensional array of rands in [0,1]
        # param[2] is the sigma of the param
        pos = [[param[1] + param[2]*np.random.randn() for param in param_info]
               for i in range(nwalkers)]

    # Initialize sampler chain
    # Recall that param_info is a list of length len(d1_params)+len(d2_params)
    # There's gotta be a more elegant way of doing this.
    ndim = len(param_info)
    if run_w_pool is True:
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        lnprob,
                                        args=(run_name, param_info),
                                        pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        lnprob,
                                        args=(run_name, param_info))

    # Initiate a generator to provide the data. More about generators here:
    # https://medium.freecodecamp.org/how-and-why-you-should-use-python-generators-f6fb56650888
    print "About to run sampler"
    run = sampler.sample(pos, iterations=nsteps, storechain=True)
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
    print "About to loop over run"
    # Instantiate the generator
    for i, result in enumerate(run):
        """Enumerate returns a tuple the element and a counter.
            tuples = [t for t in enumerate(['a', 'b', 'c'])]
            counters = [c for c, l in enumerate(['a', 'b', 'c'])]
            """
        print "Got a result"
        pos, lnprobs, blob = result
        print "Step: ", i
        # print "Lnprobs: ", lnprobs
        # print "Positions: ", pos

        # Log out the new positions
        with open(chain_filename, 'a') as f:
            new_step = [np.append(pos[k], lnprobs[k]) for k in range(nwalkers)]
            # print "Adding a new step to the chain: ", new_step
            np.savetxt(f, new_step, delimiter=',')

    if run_w_pool is True:
        pool.close()
