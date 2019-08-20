## PREAMBLE and CONTEXT
I have been working to characterize two disks around stars in a young binary in the Orion Nebula cluster. Thanks to ALMA, we are now able to get meaningful molecular line emission data on protoplanetary disks in Orion (our nearest high-mass star forming region), allowing us to now explore the role that environment plays in evolution of circumstellar disks. This work became my master's thesis at Wesleyan (see Thesis_Final.pdf; it will also presumably be uploaded to [WesScholar](https://wesscholar.wesleyan.edu/etd_mas_theses/) at some point). Alternatively, see Factor et al. (2017) for a similar project.





## INTENTION
Our goal is to, in essence, figure out what's going on in the system's that we have observed. This means looking at a couple blobs of light and inferring from them the physical parameters (i.e. mass, radius, chemical abundances, and temperature structures) of the underlying disks that are responsible for those blobs of light. To do so, we use disk modeling and ray tracing code (Flaherty 2015) to generate models of these disks (with known parameters) and simulate what that disk would look like at the distance, inclination, and position angle of the observed system. We may then compare that synthetic image to the "true" image (the actual data). By trying lots of different sets of parameters, we can find the set that best recreates the data, thus telling us about the actual disks' physical structures.

This is, obviously, an incredibly over-simplified explanation of what is actually going on. For a more thorough explanation, please see my thesis.

The files contained in this repository constitute the mechanism by which the whole thing moves.



## HOW IT WORKS
Given a range of parameters and a the appropriate molecular line data, create a best-fit model of the V2434 Ori system. Each fitting method (grid search and MCMC) choose parameter combinations differently, but both take that set of parameters, and use the disk modeling code to make two model disks, convert them into synthetic sky-projected images, convert those to the visibility domain, evaluate how well they match the observational data, and repeat this process.


The meat of this process is the "trying lots of different sets of parameters" bit. I have two ways of doing that cooked into this project: a simple way (grid search) and a smart way (Markov Chain Monte Carlo, or MCMC). I haven't touched the grid search in a long time now; it was just used for exploratory analysis. Therefore, I will now ignore it from here on out; it should be easy enough to figure out if you really want to.

The other method, MCMC, is way more fun. It is built around Dan Foreman-Mackey's package [emcee](https://emcee.readthedocs.io/en/v2.2.1/) (follow that link to learn about how MCMC works if you don't yet know). The


## EXPLANATION OF SCRIPTS

Below are some descriptions of the scripts and files, ranked in approximate order of importance.


### run_driver.py
The MCMC wrapper. Holds parameter information for the MCMC run, as well as starting the actual run. Also holds 'lnprob()', the function that 'emcee' uses to evaluate each mode's fit.

### utils.py
Holds the Model class, which takes in a list of parameters as attributes and has methods to generate a model image from those parameters.

### full_run.py
Simplifies the run-starting process by just prompting the user for the important run characteristics (which molecular line and how many processors they would like to use) and then setting things up automatically.

### analysis.py
Has some useful classes that can be used for analyzing various things, including an MCMC run, an observation (i.e. raw data) (this one isn't complete yet), and a class to create pretty moment maps.

### baseline_cutoff.py
Images a visibility set while cutting out short baselines, varying that cutoff point, before evluating RMS and mean noise for each image to find where noise is minimized. This information is then entered (manually) into the data set's dictionary in constants.py.

### var_vis.py
Calculates and adds weights to the visibilities. I didn't write this, so I don't have a super deep understanding of it, but it's used in the data processing pipeline, so it's important.

### process_data.py
A data-processing pipeline, taking the original 300GB visibility-set, splitting out the relevant data, doing some coordinate transformations, and some more splits. This feeds a subdirectory structure to hold all the data for easier access (not on GitHub).

### run_params.py
Holds parameter value ranges that 'grid_search' draws on.

### tools.py
Some analysis tools. Basically wrappers and shortcuts for MIRIAD functions. There is some overlap between this and 'analysis.py' (i.e. both can make channel maps), but this is generally aimed more towards actively modifying files (for example, the invert/clean/restore process is held in here).

### constants.py
Generates some of the annoying stuff (i.e. model velocity grids, channels, etc) but has some outdated stuff in there. Hopefully I'll clean this up soon.

### grid_search.py
This is pretty much the heart of the grid search method. Grid search over parameters for a given molecular line's dataset. Loops over value ranges for molecular abundance, atmospheric temperature, temperature profile, outer radius, and has the ability to also fit for sky position, systemic velocity, inclination, and position angle, for each disk separately. Each iteration through the loop creates a synthetic image of two model disks, converts them to the visibility domain, and calculates a chi-squared value to check how good a fit that combination of parameters yields.

### push_commit.py
A simple script to push code to GitHub, bc lazy.








## The Flow: Data Processing

The whole characterization process (i.e. the goal of this work) can be broken down into three main phases: data cleaning, modeling, and analysis. Again, the intricacies of this are somewhat better explained in my thesis, but this will be a hand-wavey approximation of it.

## Cleaning
Save for noise-characterizations, the whole data reduction process is automated out of the run_full_pipeline() function in process_data.py. Since the original ALMA visibilities have a ton of stuff we don't need (four spectral windows for a ton of different observing fields), we begin by pulling out our specific field (OrionField4) and choosing a spectral line to work with (HCO+, HCN, CO, or CS), cutting the original 300GB structure down to just a couple GB. Continuum-subtraction and a velocity-coordinate transformation (from topocentric to LSR) are run through CASA, and the 51 central channels around the actual signal from the disk are split out from the ~3800 total channels. We may then use baseline_cutoff.py to characterize the noise in each molecular line's data as a function of minimum baseline length, and if it is found that noise is minimized by removing baselines shorter than some observed length, this cut can be made in the last step of run_full_pipeline(). This leaves us with a directory called data, and subdirs for each molecular line containing the various data manifestations.

## Modeling: MCMC
The MCMC fitting used here is a heavily modified variant on [Cail Dailey's work with MCMC](https://github.com/cailmdaley/astrocail). An MCMC run is executed through a combination of run_driver.py and utils.py. The actual Markov Chain Monte Carlo is executed with the emcee Python package ([Foreman-Mackey et al, 2013](https://arxiv.org/pdf/1202.3665.pdf)).

An MCMC run is initiated from full_run.py. Once executed, run_driver.py/run_emcee() is called. This function sets up and executes the whole run.





run_emcee() calls the actual emcee package, setting up and executing its EnsembleSampler() and sample() functionality, then logging out the resulting steps to the chain csv. Those emcee functions use the lnprob function described above to evaluate steps.

Once this chain is built, run_emcee.py's main() function becomes useful, including the walker evolution plotter and the corner-plot maker.







## The Flow: Grid Search
A grid search is initiated by running full_run.py, which doesn't do too much more than just calling fullRun() from grid_search.py and providing it with run parameters (basically, lists of the values that we want to query for each fit param) from run_params.py.

The heart of fullRun() (and this whole thing) is the grid_search() monster function. This runs a grid search over one disk, holding the other constant (the assumption here is that the two disks are fully independent, and thus that the best fit model for one should be found regardless of what values we use for the other). It does this by iterating over a whole bunch of parameters, and at each step creating a set of synthetic visibilities from a sum of models of each individual disk. A chi-squared value is calculated. Information about this step is stored in two places: one monster dataframe that holds all the parameter values for each step, as well as raw and reduced chi-squared values, as well as a n_params-dimensional numpy matrix that basically defines a chi-squared space whose dimensions are each fit parameter. I don't think I actually do anything with this structure, but it might be useful for plotting.

After finding the best fit values, functions from analysis.py may be called to make visualize/interpret the results, most notably plot_gridSearch_log() (which plots numberlines of each parameter, the values of that parameter that were queried, and which value was the best); plot_model_and_data(), which plots a triptych of the original data file (from which the chi-squared values were computed), the best-fit model image, and residuals between the two; and plot_param_degeneracies(), which plots a slice of chi-squared space in two dimensions. This is useful to get a little insight into how parameters are changing relative to one another (i.e. molecular abundance and atmospheric temperature are likely to be fairly degenerate, as both are an easy way to quickly boost signal).

Scattered throughout this whole process are calls to tools.py. This script contains a bunch of variously-useful functions. Most of these are just Python wrappers for Miriad and CASA commands (cleaning, simple imaging, quick spectra, image statistics, and so on); the remainders are Python wrappers of Bash commands (remove files/directories or check if they exist). There is some ambiguity as to what belongs in tools and what belongs in analysis, but I think they're pretty well sorted right now.




## CONSTANTS, NAMES, and LAYOUT
One thing that has kinda plagued this whole process is global variables. constants.py is kinda unavoidable, but keeping track of the different path names and so on is a nasty business. Here is an attempt at a summary of them:
* run_name, run_path (in run_driver, four_line_run_driver): the name of the run (i.e. nov13-2), and the path from jonas/modeling/ to a specific run's directory, of the form './mcmc_runs/[run_name]/'. Called when initiating an MCMC object and in mcmc.run_emcee().
  * Note that, in a stroke of genius, I made this different from MCMC.MCMCrun.runpath, which is defined as run_path + [run_name, i.e. today]
* modelfiles_path (fitting.Model())





## THOUGHTS and COMMENTS

The fitting assumes that the two disks are not interacting, which may be a bad assumption. Their angular separation corresponds to a distance between the two stars of just ~400 AU, which would be well within their realm of interaction if the line connecting them is perpendicular to our line of sight, although we don't know their relative z-axis distances from us. However, right now the grid search is refusing to settle well, and it's looking like the MCMC runs are as well. Hopefully the four line fit will shed some light on things.

Grid search is mostly done, MCMC is not but is at least vaguely functional right now.
