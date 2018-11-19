## PREAMBLE and CONTEXT
I have been working to characterize two disks around stars in a young binary in the Orion Nebula cluster. Thanks to ALMA, we are now able to get meaningful molecular line emission data on protoplanetary disks in Orion (our nearest high-mass star forming region), allowing us to now probe explore the role that environment plays in evolution of circum-stellar disks. For more, see Factor et al (2017).




## INTENTION
Using disk modelling and ray tracing code (Flaherty 2015), I am working to characterize my two disks using two fitting methods. I have started off with a fairly straightforward grid search, setting up high-dimensional grids of disk parameters (disk radius, characteristic temperatures, molecular abundances, and so on), and loop over those values, using a chi-squared value as a goodness-of-fit metric and observing which results in the best outcome. I have followed that up with a Markov Chain Monte Carlo (MCMC) fitting method (Foreman-Mackey 2012), which more effectively characterizes parameter space and offers characterizations of likelihood distributions for each parameter. The following files constitute the mechanism by which the whole thing moves.


## HOW IT WORKS
Given a range of parameters and a the appropriate molecular line data, create a best-fit model of the V2434 Ori system.


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


## THOUGHTS and COMMENTS

The fitting assumes that the two disks are not interacting, which may be a bad assumption. Their angular separation corresponds to a distance between the two stars of just ~400 AU, which would be well within the realm of interaction, although we don't know their relative z-axis distances from us.
