"""
The final scripts to make thesis plots.

Would be nice to have these labeled some other way, but heck.
"""


from analysis import Figure
from tools import plot_pv_diagram_casa
from mcmc import MCMCrun
from baseline_cutoff import run_noise_analysis


###~~~~~~~~~~~~~~~~~~~~~~###
# Section 1
###~~~~~~~~~~~~~~~~~~~~~~###

def fig_113():
    Figure('data/hco/hco-short110.fits', moment=1, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment1_hco-data', title=None)

    Figure('data/hcn/hcn-short80.fits', moment=1, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment1_hcn-data', title=None, plot_bf_ellipses=False)

    Figure('data/co/co-short60.fits', moment=1, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment1_co-data', title=None)

    Figure('data/cs/cs.fits', moment=1, remove_bg=True, save=True, image_outpath='../Thesis/Figures/moment1_cs-data', title=None)



###~~~~~~~~~~~~~~~~~~~~~~
# Section 3
###~~~~~~~~~~~~~~~~~~~~~~

# Noise Profiles
#~~~~~~~~~~~~~~~~~~~
def fig_31():
    run_noise_analysis('hco', save_to_thesis=True)

def fig_32():
    run_noise_analysis('hcn', save_to_thesis=True)

def fig_33():
    run_noise_analysis('co', save_to_thesis=True)

def fig_34():
    run_noise_analysis('cs', save_to_thesis=True)
#~~~~~~~~~~~~~~~~


def fig_35():
    fig35_co = Figure(['data/co/co.fits', 'data/co/co-short60.fits'], moment=0, remove_bg=True, save=True,
                      image_outpath='../Thesis/Figures/moment0_co-baselinecuts', title='CO Moment 0 Maps')

def fig_36():
    fig35_co = Figure(['data/co/co.fits', 'data/co/co-short60.fits'], moment=1, remove_bg=True, save=True,
                      image_outpath='../Thesis/Figures/moment1_co-baselinecuts', title='CO Moment 1 Maps')


def fig_37():
    """
    I don't think this is quite right. The real plots use CASAs interactive PV
    cutter and then somehow plot the results.
    """
    plot_pv_diagram_casa()



###~~~~~~~~~~~~~~~~~~~~~~###
# Section 4
###~~~~~~~~~~~~~~~~~~~~~~###

def fig_42():
    run = MCMCrun('mcmc_runs/april9-hco/', 'april9-hco', burn_in=50)
    run.corner(save=True, save_to_thesis=True)


def fig_43():
    run = MCMCrun('mcmc_runs/april9-hco/', 'april9-hco', burn_in=50)
    run.DMR_images(save=True, save_to_thesis=True)


# If we want to add DMR moment maps
def fig_4x():
    # run = MCMCrun('mcmc_runs/april9-hco/', 'april9-hco', burn_in=50)
    # run.make_best_fits()
    # This path would require that we be on the right machine. Could just ask
    # user to be sure to have made BF fits already.
    Figure(['data/hco/hco-short110.fits',
            'mcmc_runs'], moment=1, remove_bg=True, save=True,
            image_outpath='../Thesis/Figures/moment1_co-baselinecuts', title='CO Moment 1 Maps')
    return None


def fig_44():
    Figure('data/hcn/hcn-short80.fits', moment=1, remove_bg=True, save=True,
            image_outpath='../Thesis/Figures/moment1_co-baselinecuts', title='HCN Moment 1',
            plot_bf_ellipses=True)


# Note that these won't run on iorek; move to sirius
def fig_45():
    run = MCMCrun('mcmc_runs/april9-hcn/', 'april9-hcn', burn_in=50)
    run.corner(save=True, save_to_thesis=True)

def fig_46():
    run = MCMCrun('mcmc_runs/april9-hco/', 'april9-hco', burn_in=50)
    run.DMR_images(save=True, save_to_thesis=True)



def fig_47():
    run = MCMCrun('mcmc_runs/april9-co/', 'april9-hcn', burn_in=50)
    run.corner(save=True, save_to_thesis=True)

def fig_48():
    run = MCMCrun('mcmc_runs/april9-co/', 'april9-hco', burn_in=50)
    run.DMR_images(save=True, save_to_thesis=True)


def fig_49():
    hostname = sp.check_output('hostname')

    if hostname is 'iorek':
        run = MCMCrun('mcmc_runs/april9-co/', 'april9-co', burn_in=50)
        run.plot_structure(save=True, save_to_thesis=True, cmap='inferno')

        run = MCMCrun('mcmc_runs/april9-hco/', 'april9-hco', burn_in=50)
        run.plot_structure(save=True, save_to_thesis=True, cmap='inferno')

    elif hostname is 'sirius':
        run = MCMCrun('mcmc_runs/april9-hcn/', 'april9-hcn', burn_in=50)
        run.plot_structure(save=True, save_to_thesis=True, cmap='inferno')

    else:
        print('No runs on this machine')
        return None



###~~~~~~~~~~~~~~~~~~~~~~###
# Section 5
###~~~~~~~~~~~~~~~~~~~~~~###

# Maybe some diagnostic result plots, i.e. temp prof or something.





# The End
