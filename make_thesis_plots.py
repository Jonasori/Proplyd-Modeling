"""
The final scripts to make thesis plots.

Would be nice to have these labeled some other way, but heck.
"""


from analysis import Figure
from tools import plot_pv_diagram_casa
from mcmc import MCMCrun
from baseline_cutoff import fourmol_analysis


###~~~~~~~~~~~~~~~~~~~~~~###
# Section 1
###~~~~~~~~~~~~~~~~~~~~~~###

def chap1_data_presentation(cmap='RdBu'):
    Figure('data/hco/hco-short110.fits', moment=1, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment1_hco-data', title=None, cmap=cmap)

    Figure('data/hcn/hcn-short80.fits', moment=1, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment1_hcn-data', title=None, cmap=cmap)

    Figure('data/co/co-short60.fits', moment=1, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment1_co-data', title=None, cmap=cmap)

    Figure('data/cs/cs.fits', moment=1, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment1_cs-data', title=None, cmap=cmap)



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
    Figure(['data/hco/hco.fits', 'data/hco/hco-short110.fits'], moment=1, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment1_hco-baselines', title=None, cmap=cmap)


    # Figure(['data/hcn/hcn.fits', 'data/hcn/hcn-short80.fits'], moment=0, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment0_hcn-baselines', title=None, cmap=cmap)
    Figure(['data/hcn/hcn.fits', 'data/hcn/hcn-short80.fits'], moment=1, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment1_hcn-baselines', title=None, cmap=cmap)


    # Figure(['data/co/co.fits', 'data/co/co-short60.fits'], moment=0, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment0_co-baselines', title=None, cmap=cmap)
    Figure(['data/co/co.fits', 'data/co/co-short60.fits'], moment=1, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment1_co-baselines', title=None, cmap=cmap)


    Figure('data/cs/cs.fits', moment=1, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment1_cs-baselines', title=None, cmap=cmap)


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
            image_outpath='../Thesis/Figures/moment1_co-baselinecuts', title='HCN Moment 1',
            plot_bf_ellipses=True)



def chap4_co_results():
    hostname = sp.check_output('hostname')
    if hostname is 'iorek':
        run = MCMCrun('mcmc_runs/april9-co/', 'april9-co', burn_in=50)
        run.posteriors(save=True, save_to_thesis=True)
        run.DMR_images(save=True, save_to_thesis=True)
    else:
        return "CO has to be run on iorek (or wherever the best run was done)"


def chap4_hco_results():
    hostname = sp.check_output('hostname')
    if hostname is 'iorek':
        run = MCMCrun('mcmc_runs/april9-hco/', 'april9-hco', burn_in=50)
        run.posteriors(save=True, save_to_thesis=True)
        run.DMR_images(save=True, save_to_thesis=True)
    else:
        return "HCO has to be run on iorek (or wherever the best run was done)"


def chap4_hcn_results():
    hostname = sp.check_output('hostname')
    if hostname is 'sirius':
        run = MCMCrun('mcmc_runs/april9-hcn/', 'april9-hcn', burn_in=50)
        run.posteriors(save=True, save_to_thesis=True)
        run.DMR_images(save=True, save_to_thesis=True)
    else:
        return "HCN has to be run on sirius (or wherever the best run was done)"






###~~~~~~~~~~~~~~~~~~~~~~###
# Section 5
###~~~~~~~~~~~~~~~~~~~~~~###

# Maybe some diagnostic result plots, i.e. temp prof or something.





# The End
