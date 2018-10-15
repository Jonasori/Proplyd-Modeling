import os
import re
import argparse
import numpy as np
import subprocess as sp
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.pylab import *
from matplotlib.ticker import *
from matplotlib.pylab import figure
from matplotlib.patches import Ellipse as ellipse
from astropy.visualization import astropy_mpl_style
from constants import lines, get_data_path, obs_stuff, offsets, get_data_path
plt.style.use(astropy_mpl_style)



def plot_fits(image_path, mol='hco',
              crop_arcsec=2, nchans_to_cut=12,
              cmap='magma', inset_cbar=True,
              save=True, show=True, save_to_results=True):
    """
    Plot some fits image data.

    The cropping currently assumes a square image. That could be easily
    fixed by just adding y_center, y_min, and y_max and putting them in the
    imshow() call.
    Args:
        image_path (str): full path, including filetype, to image.
        crop_arcsec (float): How many arcseconds from 0 should the axis limits be set?
        nchans_to_cut (int): cut n/2 chans off the front and end
        cmap (str): colormap to use. Magma, copper, afmhot, CMRmap, CMRmap(_r) are nice

    Known Bugs:
        - Some values of n_chans_to_cut give a weird error. I don't really wanna
            figure that out right now

    To Do:
        - Maybe do horizontal layout for better screenshots for Evernote.
    """
    # Get the data
    dataPath = get_data_path(mol, short_vis_only=True)
    real_data  = fits.getdata(dataPath + '.fits', ext=0).squeeze()

    image_data = fits.getdata(image_path, ext=0).squeeze()
    header     = fits.getheader(image_path, ext=0)

    # Get the min and max flux vals
    # Note that this is normalzing to the real data.
    vmin = np.nanmin(real_data)
    vmax = np.nanmax(real_data)

    # Add some crosses to show where the disks should be centered (arcsec)
    # offsets_dA, offsets_dB = [-0.0298, 0.072], [-1.0456, -0.1879]
    offsets_dA, offsets_dB = offsets[0], offsets[1]

    # Beam stuff
    add_beam = True if 'bmaj' in header else False
    if add_beam is True:
        bmin = header['bmin'] * 3600.
        bmaj = header['bmaj'] * 3600.
        bpa = header['bpa']

    # Cropping: How many arcsecs radius should it be
    x_center = int(np.floor(image_data.shape[1]/2))
    # If zero is entered, just don't do any cropping and show full image.
    if crop_arcsec == 0:
        crop_arcsec = 256 * 0.045

    crop_pix = int(crop_arcsec / 0.045)
    xmin, xmax = x_center - crop_pix, x_center + crop_pix

    # Set up velocities:
    chanstep_vel = header['CDELT3'] * 1e-3
    # Reference pix value - chanstep * reference pix number
    chan0_vel = (header['CRVAL3'] * 1e-3) - header['CRPIX3'] * chanstep_vel

    # Cut half of nchans_to_cut out of the front end
    chan_offset = int(np.floor(nchans_to_cut/2))
    nchans = image_data.shape[0] - nchans_to_cut

    # I think these are labeled backwards, but whatever.
    n_rows = int(np.floor(np.sqrt(nchans)))
    n_cols = int(np.ceil(np.sqrt(nchans)))
    chan_offset = 4

    fig = figure(figsize=[n_rows, n_cols])

    with plt.xkcd():
        # Add the actual data
        for i in range(nchans):
            chan = i + int(np.floor(nchans_to_cut/2))
            velocity = str(round(chan0_vel + chan * chanstep_vel, 2))
            # i+1 because plt indexes from 1
            ax = fig.add_subplot(n_cols, n_rows, i+1)
            ax.grid(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.text(0, -0.8 * crop_arcsec, velocity + ' km/s',
                    fontsize=6, color='w',
                    horizontalalignment='center', verticalalignment='center')

            if i == n_rows * (n_cols - 1) and add_beam is True:
                el = ellipse(xy=[0.8 * crop_arcsec, 0.8*crop_arcsec],
                             width=bmin, height=bmaj, angle=-bpa,
                             fc='k', ec='w', fill=False, hatch='////////')
                ax.add_artist(el)

            cmaps = imshow(image_data[i + chan_offset][xmin:xmax, xmin:xmax],
                          cmap=cmap, vmin=vmin, vmax=vmax,
                          extent=(crop_arcsec, -crop_arcsec,
                                  crop_arcsec, -crop_arcsec))

            ax.plot(offsets_dA[0], offsets_dA[1], '+g')
            ax.plot(offsets_dB[0], offsets_dB[1], '+g')


        # Make the colorbar
        # Want it to be in the gap left at the end of the channels?
        if inset_cbar is True:
            plt.tight_layout()
            fig.subplots_adjust(wspace=0.1, hspace=0.1)
            cax = plt.axes([0.55, 0.08, 0.38, 0.07])
            cbar = colorbar(cmaps, cax=cax, orientation='horizontal')
            cbar.set_label('Jy/beam',labelpad=-12,fontsize=12, weight='bold')
            cbar.set_ticks([vmin, vmax])


        # If not, put it on top.
        else:
            plt.tight_layout()
            fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9)
            cax = plt.axes([0.1, 0.95, 0.81, 0.025])
            cbar = colorbar(cmaps, cax=cax, orientation='horizontal')
            cbar.set_label('Jy/beam',labelpad=-12,fontsize=12, weight='bold')
            cbar.set_ticks([vmin, vmax])


        # Want to save? Want to show?
        if save is True:
            if save_to_results is True:
                resultsPath = '/Volumes/disks/jonas/freshStart/modeling/results/'
                run_date = image_path.split('/')[-2]
                outpath = resultsPath + run_date + '_image.png'
            else:
                run_path = image_path.split('.')[0]
                outpath = run_path + '_image.png'

            plt.savefig(outpath, dpi=200)
            print "Image saved to" + outpath + '_image.png'

        if show is True:
            plt.show(block=False)
        plt.gca()
