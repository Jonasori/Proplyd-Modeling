## PREAMBLE and CONTEXT
I have been working to characterize two disks around stars in a young binary in the Orion Nebula cluster. Thanks to ALMA, we are now able to get meaningful molecular line emission data on protoplanetary disks in Orion (our nearest high-mass star forming region), allowing us to now probe explore the role that environment plays in evolution of circum-stellar disks. For more, see Factor et al (2017).




## INTENTION
Using disk modelling and ray tracing code (Flaherty 2015), I am working to characterize my two disks using two fitting methods. I have started off with a fairly straightforward grid search, setting up high-dimensional grids of disk parameters (disk radius, characteristic temperatures, molecular abundances, and so on), and loop over those values, using a chi-squared value as a goodness-of-fit metric and observing which results in the best outcome. I have followed that up with a Markov Chain Monte Carlo (MCMC) fitting method (Foreman-Mackey 2012), which more effectively characterizes parameter space and offers characterizations of likelihood distributions for each parameter. The following files constitute the mechanism by which the whole thing moves.


## HOW IT WORKS
Given a range of parameters and a the appropriate molecular line data, create a best-fit model of the V2434 Ori system. Each fitting method (grid search and MCMC) choose parameter combinations differently, but both take that set of parameters, and use the disk modeling code to make two model disks, convert them into synthetic sky-projected images, convert those to the visibility domain, evaluate how well they match the observational data, and repeat this process.


EXPLANATION OF SCRIPTS
### analysis.py
Contains fairly generic plotting and analysis tools for making channel maps in various forms and summarizing grid search runs, as well as a tool to read in the pickle'd dataframe that the grid-search makes.

### baseline_cutoff.py
Images a visibility set while cutting out short baselines, varying that cutoff point, before evluating RMS and mean noise for each image to find where noise is minimized. This information is then entered (manually) into the data set's dictionary in constants.py.

### constants.py
A place to centrally store constants and other such things. The 'lines' dictionary holds characteristic information about each molecular line I'm looking at (HCO+, HCN, CS, and CO), including which min baseline cutoff results in the lowest noise, the line's rest frequency, and which spectral window (in the original, full dataset) the line is held in.

### four-line-gridsearch.py
A script to simultaneously fit all four lines. Since we expect things like temperatures to be constant across lines, we don't want each line's individual fit to disagree on these universal values, and so fitting all at once could be helpful in solving this. Totally unimplemented right now.

### grid_search.py
Grid search over parameters for a given molecular line's dataset. Loops over value ranges for molecular abundance, atmospheric temperature, temperature profile, outer radius, and has the ability to also fit for sky position, systemic velocity, inclination, and position angle, for each disk separately. Each iteration through the loop creates a synthetic image of two model disks, converts them to the visibility domain, and calculates a chi-squared value to check how good a fit that combination of parameters yields.

### mcmc.py
One of three core files for the MCMC routine. 'mcmc.py' is responsible for setting up an MCMC object (holding information and data about the run).

### process_data.py
A data-processing pipeline, taking the original 300GB visibility-set, splitting out the relevant data, doing some coordinate transformations, and some more splits. This feeds a subdirectory structure to hold all the data for easier access (not on GitHub).

### push_commit.py
A simple script to push code to GitHub. Not super great right now I don't think

### run_driver.py
The MCMC wrapper. Holds parameter information for the MCMC run, as well as starting the actual run. Also holds 'lnprob()', the function that 'emcee' uses to evaluate each mode's fit.

### run_params.py
Holds parameter value ranges that 'grid_search' draws on.

### tools.py
Some analysis tools. Basically wrappers and shortcuts for MIRIAD functions. There is some overlap between this and 'analysis.py' (i.e. both can make channel maps), but this is generally aimed more towards actively modifying files (for example, the invert/clean/restore process is held in here).

### utils.py
Holds functions to actually build the disk models, sum them, and evaluate their chi-squared value compared to the relevant data.

### var_vis.py
Calculates and adds weights to the visibilities. I didn't write this, so I don't have a super deep understanding of it, but it's used in the data processing pipeline, so it's important.


## IGNORED FILES
These files are not included in this repo but are not described above, mostly because they're not relevant to me right now for whatever reasons (pulled from someone else's writing and currently unused, or just outdated)

channelMapMaker.py
colormaps.py (used by fitting I think?)
plot_xkcd.py
v2434or_fitting.py
plotting.py



## The Flow: Data Processing
The data pipeline is tuned pretty specifically to my directory structure. Save for noise-characterizations, the whole data reduction process is automated out of the run_full_pipeline() function in process_data.py. Since the original ALMA visibilities have a ton of stuff we don't need (four spectral windows for a ton of different sky fields), we begin by pulling out our specific field (OrionField4) and choosing a spectral line to work with (HCO+, HCN, CO, or CS), cutting the original 300GB structure down to just a couple gigs. A continuum-subtraction and a velocity-coordinate transformation (from topocentric to LSR) are run through CASA, and the 51 central channels around the actual signal from the disk are split out from the ~3800 total channels. We may then use baseline_cutoff.py to characterize the noise in each molecular line's data as a function of minimum baseline length, and if it is found that noise is minimized by removing baselines shorter than some observed length, this cut can be made in the last step of run_full_pipeline(). Running this on all four lines leaves us with a directory structure of:

                  images/ ------------ noise.png, hco_image.png, hco-short_image.png
         hcn/ ----|
         |        baseline_testing/ -- [all the test files used to create noise.png]
data/ -- |        |
         |        hcn.[fits, cm, ms, uvf, vis]
         |        hcn-short.[fits, cm, ms, uvf, vis]
         |
         hco/etc
         |
         co/etc
         |
         cs/etc

(Woops, that didn't render well on GitHub)


## The Flow: Grid Search
A grid search is initiated by running full_run.py, which doesn't do too much more than just calling fullRun() from grid_search.py and providing it with run parameters (basically, lists of the values that we want to query for each fit param) from run_params.py.

The heart of fullRun() (and this whole thing) is the grid_search() monster function. This runs a grid search over one disk, holding the other constant (the assumption here is that the two disks are fully independent, and thus that the best fit model for one should be found regardless of what values we use for the other). It does this by iterating over a whole bunch of parameters, and at each step creating a set of synthetic visibilities from a sum of models of each individual disk. A chi-squared value is calculated. Information about this step is stored in two places: one monster dataframe that holds all the parameter values for each step, as well as raw and reduced chi-squared values, as well as a n_params-dimensional numpy matrix that basically defines a chi-squared space whose dimensions are each fit parameter. I don't think I actually do anything with this structure, but it might be useful for plotting.

After finding the best fit values, functions from analysis.py may be called to make visualize/interpret the results, most notably plot_gridSearch_log() (which plots numberlines of each parameter, the values of that parameter that were queried, and which value was the best); plot_model_and_data(), which plots a triptych of the original data file (from which the chi-squared values were computed), the best-fit model image, and residuals between the two; and plot_param_degeneracies(), which plots a slice of chi-squared space in two dimensions. This is useful to get a little insight into how parameters are changing relative to one another (i.e. molecular abundance and atmospheric temperature are likely to be fairly degenerate, as both are an easy way to quickly boost signal).

Scattered throughout this whole process are calls to tools.py. This script contains a bunch of variously-useful functions. Most of these are just Python wrappers for Miriad and CASA commands (cleaning, simple imaging, quick spectra, image statistics, and so on); the remainders are Python wrappers of Bash commands (remove files/directories or check if they exist). There is some ambiguity as to what belongs in tools and what belongs in analysis, but I think they're pretty well sorted right now.


## The Flow: MCMC
The MCMC fitting used here is a heavily modified variant on [Cail Dailey's work with MCMC](https://github.com/cailmdaley/astrocail). An MCMC run is executed through a combination of mcmc.py, fitting.py, and run_driver.py. The actual Markov Chain Monte Carlo is executed with the emcee Python package ([Foreman-Mackey et al, 2013](https://arxiv.org/pdf/1202.3665.pdf)).

An MCMC run is initiated by calling mpirun -np [number of available processors] python run_driver -r. The leading command (mpirun...) tells emcee how many processors to parallelize the run across. The -r suffix triggers the run command in run_driver.py.

This file is the base from which the run is executed. It holds a bunch of stuff:
* param_dict, a dictionary that is dynamically updated in each step and which holds all the parameter values (not just those we are fitting for).
* param_info, a list of tuples describing the parameters that we will be fitting fit. Each tuple contains the parameter's name, a starting position, a sigma value defining the distribution of steps out from the starting position, and a tuple of upper and lower bounds on how far the steps can wander.
* lnprob(), a function that makes a set of synthetic visibilities, calculates a chi-squared value, and returns a log-likelihood value (-0.5 * sum(chi vals)).
* make_fits(), a function that just makes a model image from a set of visibilities.
* main(), a function that allows extra arguments (like -r) to be added to the initial call. There are a bunch of options here, but the one that's relevant now is -r, which calls run_emcee() from mcmc.py.


The mcmc.py script has two main pieces: the MCMCrun class, and the run_emcee() function. The class is basically just a wrapper that holds information about a given run, including its name, where to find the csv file with all the steps in it, how many walkers/steps it took, and so on.

run_emcee() calls the actual emcee package, setting up and executing its EnsembleSampler() and sample() functionality, then logging out the resulting steps to the chain csv. Those emcee functions use the lnprob function described above to evaluate steps.

Once this chain is built, run_emcee.py's main() function becomes useful, including the walker evolution plotter and the corner-plot maker.




## CONSTANTS, NAMES, and LAYOUT
One thing that has kinda plagued this whole process is global variables. constants.py is kinda unavoidable, but keeping track of the different path names and so on is a nasty business. Here is an attempt at a summary of them:
* run_name, run_path (in run_driver, four_line_run_driver): the name of the run (i.e. nov13-2), and the path from jonas/modeling/ to a specific run's directory, of the form './mcmc_runs/[run_name]/'. Called when initiating an MCMC object and in mcmc.run_emcee().
  * Note that, in a stroke of genius, I made this different from MCMC.MCMCrun.runpath, which is defined as run_path + [run_name, i.e. today]
* modelfiles_path (fitting.Model())





## THOUGHTS and COMMENTS

The fitting assumes that the two disks are not interacting, which may be a bad assumption. Their angular separation corresponds to a distance between the two stars of just ~400 AU, which would be well within their realm of interaction if the line connecting them is perpendicular to our line of sight, although we don't know their relative z-axis distances from us. However, right now the grid search is refusing to settle well, and it's looking like the MCMC runs are as well. Hopefully the four line fit will shed some light on things.

Grid search is mostly done, MCMC is not but is at least vaguely functional right now.
