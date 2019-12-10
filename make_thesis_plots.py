"""
The final scripts to make thesis (/paper) plots.

Would be nice to have these labeled some other way, but heck.
"""

import subprocess as sp
from analysis import Figure
from tools import plot_pv_diagram_casa, moment_maps
from analysis import MCMC_Analysis
from baseline_cutoff import fourmol_analysis


save_for_paper = True
if save_for_paper:
	p = '/Volumes/disks/jonas/v2434ori_paper/Figures/'
#     ext = '.pdf'
else:
	p = '/Volumes/disks/jonas/Thesis/Figures/'
#     ext = '.png'

run_date = 'april9'
ext = '.pdf'

best_co = 'april6-co'
best_hco = 'april9-hco'
best_hcn = 'oct30-hcn'
best_multi = 'april9-multi'

###~~~~~~~~~~~~~~~~~~~~~~###
# Section 1
###~~~~~~~~~~~~~~~~~~~~~~###

def chap1_data_presentation(cmap='RdBu'):
    # TODO: Broken bc constants no longer defined (analysis.py/line 1473)
    Figure('data/hco/hco-short110.fits', moment=1, remove_bg=True, save=True, image_outpath=p + 'moment1_hco-data', title=None, cmap=cmap, ext=ext)

    Figure('data/hcn/hcn-short80.fits', moment=1, remove_bg=True, save=True, image_outpath=p + 'moment1_hcn-data', title=None, cmap=cmap, ext=ext)

    Figure('data/co/co-short60.fits', moment=1, remove_bg=True, save=True, image_outpath=p + 'moment1_co-data', title=None, cmap=cmap, ext=ext)

    Figure('data/cs/cs.fits', moment=1, remove_bg=True, save=True, image_outpath=p + 'moment1_cs-data', title=None, cmap=cmap, ext=ext)



###~~~~~~~~~~~~~~~~~~~~~~
# Section 3
###~~~~~~~~~~~~~~~~~~~~~~

# Noise Profiles
#~~~~~~~~~~~~~~~~~~~
def chap3_baseline_cuts():
    fourmol_analysis(cmap='twilight_shifted', save=True)
#~~~~~~~~~~~~~~~~

def chap3_baseline_momentmaps(cmap='RdBu'):
    # Figure(['data/hco/hco.fits', 'data/hco/hco-short110.fits'], moment=0, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment0_hco-baselines', title=None, cmap=cmap)
    Figure(['data/hco/hco.fits', 'data/hco/hco-short110.fits'], moment=1, remove_bg=True, save=True, image_outpath=p + 'moment1_hco-baselines', title=None, cmap=cmap, ext=ext)


    # Figure(['data/hcn/hcn.fits', 'data/hcn/hcn-short80.fits'], moment=0, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment0_hcn-baselines', title=None, cmap=cmap)
    Figure(['data/hcn/hcn.fits', 'data/hcn/hcn-short80.fits'], moment=1, remove_bg=True, save=True, image_outpath=p + 'moment1_hcn-baselines', title=None, cmap=cmap, ext=ext)


    # Figure(['data/co/co.fits', 'data/co/co-short60.fits'], moment=0, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment0_co-baselines', title=None, cmap=cmap)
    Figure(['data/co/co.fits', 'data/co/co-short60.fits'], moment=1, remove_bg=True, save=True, image_outpath=p + 'moment1_co-baselines', title=None, cmap=cmap, ext=ext)


    Figure('data/cs/cs.fits', moment=1, remove_bg=True, save=True, image_outpath=p + 'moment1_cs-baselines', title=None, cmap=cmap, ext=ext)


def chap3_pvd():
    """
    I don't think this is quite right. The real plots use CASAs interactive PV
    cutter and then somehow plot the results.
    """
    yn = input("Have you already CASA'ed a PVD fits? If not, do so now because \
               this function just plots fits files. See tools.py/plot_pv_diagram_fits() \
               docstring for more info.\n[y/n]: ")
    if yn.lower() is 'y':
        plot_pv_diagram_fits()




###~~~~~~~~~~~~~~~~~~~~~~###
# Section 4
###~~~~~~~~~~~~~~~~~~~~~~###


# Plot out the HCN moment maps with ellipses overlaid
def chap4_hcn_ellipses():
    Figure('data/hcn/hcn-short80.fits', moment=1, remove_bg=True, save=True,
            image_outpath=p + 'moment1_hcn-ellipses', # title='Visualizing HCN Model Radius Fits',
            plot_bf_ellipses=True)



def chap4_co_results():
    run = MCMC_Analysis(best_co, burn_in=50)
    run.posteriors(save=True, save_to_thesis=True)
    run.plot_structure(save=True, save_to_thesis=True)
    # moment_maps(im_path='mcmc_runs/april9-co/model_files/april9-co_bestFit', moment=0)
    moment_maps(im_path='./mcmc_runs/' + best_co + '/model_files/' + best_co + '_bestFit_resid', moment=0)
    run.DMR_images(save=True, save_to_thesis=True)
    dmr_maps = Figure(['data/co/co-short60.fits',
                       'mcmc_runs/' + best_co + '/model_files/' + best_co + '_bestFit.fits',
                       'mcmc_runs/' + best_co + '/model_files/' + best_co + '_bestFit_resid.fits'],
                      image_outpath=p + 'DMRmoments_co.png', save=True)


def chap4_hco_results():
    run = MCMC_Analysis(best_hco, burn_in=50)
    run.posteriors(save=True, save_to_thesis=True)
    run.plot_structure(save=True, save_to_thesis=True)
    # moment_maps(im_path='mcmc_runs/april9-hco/model_files/april9-hco_bestFit', moment=0)
    moment_maps(im_path='mcmc_runs/' + best_hco + '/model_files/' + best_hco + '_bestFit_resid', moment=0)
    run.DMR_images(save=True, save_to_thesis=True)
    dmr_maps = Figure(['data/hco/hco-short110.fits',
                       'mcmc_runs/' + best_hco + '/model_files/' + best_hco + '_bestFit.fits',
                       'mcmc_runs/' + best_hco + '/model_files/' + best_hco + '_bestFit_resid.fits'],
                      image_outpath=p + 'DMRmoments_hco' + ext, save=True)

def chap4_hcn_results(remove_large_r=False):
    if remove_large_r:
        run.groomed = run.groomed[run.groomed['r_out_B'] < 250]
        run.main = run.main[run.main['r_out_B'] > 250]
        run.get_fit_stats()

    run = MCMC_Analysis(best_hcn, burn_in=50)
    run.posteriors(save=True, save_to_thesis=True)
    run.plot_structure(save=True, save_to_thesis=True)
    # moment_maps(im_path='mcmc_runs/april9-hcn/model_files/april9-hcn_bestFit', moment=0)
    moment_maps(im_path='mcmc_runs/' + best_hcn + '/model_files/' + best_hcn + '_bestFit_resid', moment=0)
    run.DMR_images(save=True, save_to_thesis=True)
    dmr_maps = Figure(['data/hcn/hcn-short80.fits',
                       'mcmc_runs/' + best_hcn + '/model_files/' + best_hcn + '_bestFit.fits',
                       'mcmc_runs/' + best_hcn + '/model_files/' + best_hcn + '_bestFit_resid.fits'],
                      image_outpath=p + 'DMRmoments_hcn' + ext, save=True)





# Could just make these three into one function with a mol argument
def chap4_get_bftab_hco():
        run = MCMC_Analysis(best_hco, burn_in=50)
        return run.fit_stats


def chap4_get_bftab_co():
    run = MCMC_Analysis(best_co, burn_in=50)
    return run.fit_stats


def chap4_get_bftab_hcn(remove_large_r = False):

    run = MCMC_Analysis(best_hcn, burn_in=50)

    if remove_large_r:
        run.groomed = run.groomed[run.groomed['r_out_B'] < 250]
        run.main = run.main[run.main['r_out_B'] < 250]
        run.get_fit_stats()

    best_fits = run.fit_stats['best fit']

    return run.fit_stats












###~~~~~~~~~~~~~~~~~~~~~~###
# Section 5
###~~~~~~~~~~~~~~~~~~~~~~###

# Maybe some diagnostic result plots, i.e. temp prof or something.





# The End
