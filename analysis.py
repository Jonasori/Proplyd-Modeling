"""
Functions to analyze and plot output from a gridSearch run.

IDEA: Add molecular line name to the data (and model) headers. Would save a lot of
        string parsing hassle. Would've been nice to think of this a couple
        months ago.

To-Do: In GridSearch_Run.param_degeneracies(), choose which slice of the
       non-plotted params we're looking through. Right now it's not doing that,
       so it's not showing the best-fit point in p-space.
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import numpy.ma as ma
import subprocess as sp
import matplotlib.pyplot as plt
# import matplotlib.pyplot.RcParams
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects

from pathlib2 import Path
from astropy.io import fits
from matplotlib.pylab import *
from matplotlib.ticker import *
from matplotlib.pylab import figure
from matplotlib.patches import Ellipse
from astropy.visualization import astropy_mpl_style
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, LinearLocator, AutoMinorLocator

import sys
sys.version

from mcmc import MCMCrun
from tools import imstat, imstat_single, pipe, moment_maps
from constants import lines, get_data_path, obs_stuff, offsets, get_data_path, mol
import constants

import astropy.units as u
from astropy.utils import data
# https://media.readthedocs.org/pdf/spectral-cube/latest/spectral-cube.pdf
# from spectral_cube import SpectralCube

os.chdir('/Volumes/disks/jonas/modeling')
Path.cwd()

sns.set_style('white')
plt.style.use(astropy_mpl_style)
matplotlib.rcParams['font.sans-serif'] = 'Times'
matplotlib.rcParams['font.family'] = 'serif'


Path.cwd()





class Figure:
    """
    Make publication-quality plots:
    - zeroth- and first-moment maps
    - Disk structure

    Note that there are some big assumptions about file names/structures made here:
    1. Images are saved to ../Thesis/Figures/. If this doesn't exist, then trouble.
    2. Assumes that the name of the molecular line of the observation is in the
        fits file name. Without this (or with conflicting ones), it won't be able
        to determine which line a fits file is represents.

    Some nice colormaps:
        RdBu, cividis, Spectral
    """

    # Set seaborn plot styles and color pallete
    sns.set_style("ticks", {"xtick.direction": "in", "ytick.direction": "in"})
    sns.set_context("paper")

    def __init__(self, paths, make_plot=True, save=False, moment=0, remove_bg=True,
                 texts=None, title=None, image_outpath=None, export_fits_mom=False,
                 plot_bf_ellipses=False, cmap='RdBu'):
        """
        Make a nice image from a fits file.

        Args:
            paths (list or str): paths to the fits file to be used, including .fits
            make_plot (bool): Whether or not to actually make the plots at all.
            save (bool): Whether or not to save the image.
                         If False, it will be shown instead.
            moment (0 or 1): Which moment map to make.
            remove_bg (bool): Whether or not to generate a mask to white out
                              pixels with intensities less than n*sigma
            texts (str): Idk. Thinking about getting rid of this.
            title (str): Title for the whole plot.
        """
        self.title = title
        self.moment = moment
        self.export_fits_mom = export_fits_mom
        self.remove_bg = remove_bg
        self.paths = np.array(([paths]) if type(paths) is str else paths)
        self.plot_bf_ellipses = plot_bf_ellipses
        self.cmap = cmap
        # This is gross but functional. The break is important.
        self.mols = []
        mols = ['hco', 'hcn', 'cs', 'co']
        for path in self.paths:
            for mol in mols:
                if mol in path:
                    self.mols.append(mol)
                    break


        self.outpath = '../Thesis/Figures/m{}-map_{}.png'.format(moment,
                                                                '-'.join(self.mols),
                                                                dpi=300)
        self.outpath = self.outpath if image_outpath is None else image_outpath

        # Clear any pre existing figures, then create figure
        plt.close()
        if make_plot:
            self.rows, self.columns = (1, len(self.paths))
            self.fig, self.axes = plt.subplots(self.rows, self.columns,
                                               figsize=(
                                                   # 11.6/2 * self.columns, 6.5*self.rows),
                                                   7*self.columns, 6.5*self.rows),
                                               sharex=False, sharey=True, squeeze=False)
            plt.subplots_adjust(wspace=-0.0)

            texts = np.array([texts], dtype=object) if type(texts) is str \
                else np.array(texts, dtype=object)
            # What is this doing?
            if type(texts.flatten()[0]) is not float:
                texts = texts.flatten()

            # Get global vmin and vmax
            vmin, vmax = 1e10, -1e10
            for p in self.paths:
                d = fits.getdata(p)
                vmin = np.nanmin(d) if vmin > np.nanmin(d) else vmin
                vmax = np.nanmax(d) if vmax < np.nanmax(d) else vmax
            self.vmin, self.vmax = vmin, vmax

            # Populate the stuff
            print((self.paths, self.mols))
            for ax, path, mol in zip(self.axes.flatten(), self.paths, self.mols):
                self.get_fits(path, mol)
                self.make_axis(ax)
                self.fill_axis(ax, mol)

            if save:
                # if image_outpath:
                #     plt.savefig(image_outpath, dpi=200)
                #     print("Saved image to {}.png".format(image_outpath))
                # else:

                plt.savefig(self.outpath, dpi=200)
                print("Saved image to {}".format(self.outpath))
            else:
                plt.show(block=False)


    def get_fits_manually(self, path, mol):
        """Make moment maps by hand. Should not be used."""
        fits_file = fits.open(path)
        self.head = fits_file[0].header
        self.data = fits_file[0].data.squeeze()

        # Read in header spatial info to create ra
        nx, ny, nv = self.head['NAXIS1'], self.head['NAXIS2'], self.head['NAXIS3']
        xpix, ypix = self.head['CRPIX1'], self.head['CRPIX2']
        xval, yval = self.head['CRVAL1'], self.head['CRVAL2']
        self.xdelt, self.ydelt = self.head['CDELT1'], self.head['CDELT2']

        # Convert from degrees to arcsecs
        self.ra_offset = np.array(
            ((np.arange(nx) - xpix + 1) * self.xdelt) * 3600)
        self.dec_offset = np.array(
            ((np.arange(ny) - ypix + 1) * self.ydelt) * 3600)

        # Assumes we have channels (i.e. nv > 1)
        try:
            self.rms = imstat(path.split('.')[-2])[1] * nv
        except sp.CalledProcessError:
            self.rms = 0
        # Decide which moment map to make.
        # www.atnf.csiro.au/people/Tobias.Westmeier/tools_hihelpers.php#moments
        if self.moment == 0:
            # Integrate intensity over pixels.
            self.im = np.sum(self.data, axis=0)
            if self.remove_bg:
                self.im = ma.masked_where(self.im < self.rms, self.im, copy=True)

        elif self.moment == 1:
            self.im = np.zeros((nx, ny))

            vsys = constants.obs_stuff(mol)[0]
            obsv = constants.obs_stuff(mol)[3] - vsys[0]

            # There must be a way to do this with array ops.
            # obsv = obsv.reshape([len(obsv)]).shape
            # self.im = np.sum(self.data * obsv, axis=0)/np.sum(self.data, axis=0)

            for x in range(nx):
                for y in range(ny):
                    # I think this is doing good stuff.
                    self.im[x, y] = np.sum(self.data[:, x, y] * obsv)
                    # self.im[x, y] = np.sum(self.data[:, x, y] * obsv)/np.sum(self.data[:, x, y])

        if self.remove_bg:
            self.im = ma.masked_where(abs(self.im) < self.rms, self.im, copy=True)

        if self.export_fits_mom:
            fits_out = fits.PrimaryHDU()
            fits_out.header = self.head
            fits_out.data = self.im
            modeling = '/Volumes/disks/jonas/modeling/'
            outpath = modeling + 'data/{}/{}-moment{}.fits'.format(mol, mol,
                                                                   self.moment)
            fits_out.writeto(outpath, overwrite=True)
            print(("Wrote out moment {} fits file to {}".format(self.moment,
                                                               outpath)))
            # change units to micro Jy
        # self.im *= 1e6
        # self.rms *= 1e6


    def get_fits(self, path, mol):
        """Docstring."""
        fits_file = fits.open(path)
        self.head = fits_file[0].header
        self.data = fits_file[0].data.squeeze()

        # Read in header spatial info to create ra
        nx, ny, nv = self.head['NAXIS1'], self.head['NAXIS2'], self.head['NAXIS3']
        xpix, ypix = self.head['CRPIX1'], self.head['CRPIX2']
        xval, yval = self.head['CRVAL1'], self.head['CRVAL2']
        self.xdelt, self.ydelt = self.head['CDELT1'], self.head['CDELT2']

        # Convert from degrees to arcsecs
        self.ra_offset = np.array(
            ((np.arange(nx) - xpix + 1) * self.xdelt) * 3600)
        self.dec_offset = np.array(
            ((np.arange(ny) - ypix + 1) * self.ydelt) * 3600)

        # Check if we're looking at 2- or 3-dimensional data
        if len(self.data.shape) == 3:
            # Make some moment maps. Make both maps and just choose which data to use.
            momentmap_basepath = path.split('.')[-2]
            moment_maps(momentmap_basepath, clip_val=0, moment=0)
            self.rms = imstat_single(momentmap_basepath + '.moment0')[1]

            moment_maps(momentmap_basepath, clip_val=self.rms, moment=0)
            self.im = fits.getdata(momentmap_basepath + '.moment0.fits').squeeze()

            if self.moment == 1:
                moment_maps(momentmap_basepath, clip_val=self.rms, moment=1)
                self.im_mom1 = fits.getdata(momentmap_basepath + '_moment1.fits').squeeze()

        else:
            self.im = self.data
            self.rms = imstat_single(path.split('.')[-2])[1]


        if self.export_fits_mom:
            fits_out = fits.PrimaryHDU()
            fits_out.header = self.head
            data = self.im if self.moment is 0 else self.im_mom1
            fits_out.data = data[100:160, 100:180]
            modeling = '/Volumes/disks/jonas/modeling/'
            outpath = modeling + 'data/{}/{}-moment{}.fits'.format(mol, mol,
                                                                   self.moment)
            fits_out.writeto(outpath, overwrite=True)
            print(("Wrote out moment {} fits file to {}".format(self.moment,
                                                               outpath)))
            print("NOTE: ^^ That moment map was cropped (in line ~800)")
        # change units to micro Jy
        # self.im *= 1e6
        # self.rms *= 1e6


    def make_axis(self, ax):
        """Docstring."""
        # Set seaborn plot styles and color pallete
        sns.set_style("ticks",
                      {"xtick.direction": "in",
                       "ytick.direction": "in"})
        sns.set_context("talk")

        xmin = -2.0
        xmax = 2.0
        ymin = -2.0
        ymax = 2.0
        ax.set_xlim(xmax, xmin)
        ax.set_ylim(ymin, ymax)
        ax.grid(False)

        # Set x and y major and minor tics
        majorLocator = MultipleLocator(1)
        ax.xaxis.set_major_locator(majorLocator)
        ax.yaxis.set_major_locator(majorLocator)

        minorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_minor_locator(minorLocator)

        # Set x and y labels
        ax.set_xlabel(r'$\Delta \alpha$ (")', fontsize=18)
        ax.set_ylabel(r'$\Delta \delta$ (")', fontsize=18)

        # tick_labs = ['', '', '-4', '', '-2', '', '0', '', '2', '', '4', '']
        tick_labs = ['', '', '-1', '', '1', '', '']
        ax.xaxis.set_ticklabels(tick_labs, fontsize=18)
        ax.yaxis.set_ticklabels(tick_labs, fontsize=18)
        ax.tick_params(which='both', right='on', labelsize=18, direction='in')

        # Set labels depending on position in figure
        if np.where(self.axes == ax)[1] % self.columns == 0:  # left
            ax.tick_params(axis='y', labelright='off', right='on')
        elif np.where(self.axes == ax)[1] % self.columns == self.columns - 1:  # right
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft='off', labelright='on')
        else:  # middle
            ax.tick_params(axis='y', labelleft='off')
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Set physical range of colour map
        self.extent = [self.ra_offset[0], self.ra_offset[-1],
                       self.dec_offset[-1], self.dec_offset[0]]


    def fill_axis(self, ax, mol):
        """Plot image as a colour map."""

        # This is massively hacky, but works. Basically, if we're plotting as
        # moment1 map, we still want to contour with moment0 lines. So this:
        try:
            im = self.im_mom1
            cbar_lab = r'$km \, s^{-1}$'
            vmin, vmax = 7, 13

        except AttributeError:
            im = self.im
            cbar_lab = r'$Jy / beam$'
            vmax = max((-np.nanmin(im), np.nanmax(im)))
            vmin = -vmax
            vmin, vmax = np.nanmin(im), np.nanmax(im)
            vmin, vmax = self.vmin, self.vmax

        cmap = ax.imshow(im, extent=self.extent,
                         vmin=vmin, vmax=vmax,
                         cmap=self.cmap)


        if self.rms:
            cont_levs = np.arange(3, 30, 3) * self.rms
            # add residual contours if resdiual exists; otherwise, add image contours
            try:
                ax.contour(self.resid,
                           levels=cont_levs,
                           colors='k',
                           linewidths=0.75,
                           linestyles='solid')
                ax.contour(self.resid,
                           levels=-1 * np.flip(cont_levs, axis=0),
                           colors='k',
                           linewidths=0.75,
                           linestyles='dashed')
            except AttributeError:
                ax.contour(self.ra_offset, self.dec_offset, self.im,
                           colors='k',
                           levels=cont_levs,
                           linewidths=0.75,
                           linestyles='solid')
                ax.contour(self.ra_offset, self.dec_offset, self.im,
                           levels=-1 * np.flip(cont_levs, axis=0),
                           colors='k',
                           linewidths=0.75,
                           linestyles='dashed')

        # Create the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="8%", pad=0.0)
        cbar = self.fig.colorbar(cmap, ax=ax, cax=cax, orientation='horizontal')
        cbar.ax.xaxis.set_tick_params(direction='out', length=3, which='major',
                                      bottom='off', top='on', labelsize=12, pad=-2,
                                      labeltop='on', labelbottom='off')

        cbar.ax.xaxis.set_tick_params(direction='out', length=2, which='minor',
                                      bottom='off', top='on')

        if np.nanmax(self.im) > 500:
            tickmaj, tickmin = 200, 50
        elif np.nanmax(self.im) > 200:
            tickmaj, tickmin = 100, 25
        elif np.nanmax(self.im) > 100:
            tickmaj, tickmin = 50, 10
        elif np.nanmax(self.im) <= 100:
            tickmaj, tickmin = 20, 5


        # minorLocator = AutoMinorLocator(tickmaj / tickmin)
        # cbar.ax.xaxis.set_minor_locator(minorLocator)
        # cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(),
        #                         rotation=45, fontsize=18)
        # cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), fontsize=18)
        # cbar.set_ticks(np.arange(-10*tickmaj, 10*tickmaj, tickmaj))

        # Colorbar label. No idea why the location of this is so weird.
        # cbar.ax.text(0.425, 0.320, r'$\mu Jy / beam$', fontsize=12,
        # cbar.ax.text(0.425, 0.320, cbar_lab, fontsize=12,

        cbar_x, cbar_y = np.mean((vmin, vmax)), np.mean((vmin, vmax))
        cbar.ax.text(cbar_x, cbar_y, cbar_lab, fontsize=12, ha='center', va='center',
                     path_effects=[PathEffects.withStroke(linewidth=2, foreground="w")])
        # cbar.set_label(cbar_lab, fontsize=13, path_effects=[PathEffects.withStroke(linewidth=4, foreground="w")],
        #                labelpad=-5)


        # Overplot the beam ellipse
        try:
            bmin = self.head['bmin'] * 3600.
            bmaj = self.head['bmaj'] * 3600.
            bpa = self.head['bpa']

            el = Ellipse(xy=[1.5, -1.5], width=bmin, height=bmaj, angle=-bpa,
                         edgecolor='k', hatch='///', facecolor='white', zorder=10)
            ax.add_artist(el)
        except KeyError:
            print("Unable to plot beam; couldn't find header info.")

        # Plot the scale bar
        if np.where(self.axes == ax)[1][0] == 0:  # if first plot
            x, y = -0.8, -1.7        # arcsec location
            ax.plot([x, x - 400/389], [y, y], '-', linewidth=3, color='darkorange')
            ax.text(x - 0.1, y + 0.15, "400 au", fontsize=12,
                path_effects=[PathEffects.withStroke(linewidth=2, foreground="w")])

        # Plot crosses at the source positions
        (posx_A, posy_A), (posx_B, posy_B) = constants.offsets
        ax.plot([posx_A], [posy_A], '+', markersize=6,
                markeredgewidth=2, color='darkorange')
        ax.plot([posx_B], [posy_B], '+', markersize=6,
                markeredgewidth=2, color='darkorange')

        # Plot ellipses with each disk's best-fit radius and inclination:
        print('\n\n\n\nLine is {}\n\n\n\n\n'.format(mol))
        # if mol.lower() == 'hcn':
        if self.plot_bf_ellipses is True:
            print("\n\n\nAdding ellipses")
            # ax.set_xlim(-1, 2)
            # ax.set_ylim(-1, 2)
            r_A = (340/389)
            r_B1 = (380/389)
            r_B2 = (150/389)
            PA_A, PA_B = 90 - 69, 136
            incl_A, incl_B = 65, 45
            print("Note that this is manually HCN specific rn, with:")
            print("rA = {}\nrB_out = {}\nrB_in = {}\nand some PAs/incls\n\n\n\n".format(r_A, r_B1, r_B2))
            ellipse_A = Ellipse(xy=(posx_A, posy_A),
                                width=r_A, height=r_A*np.sin(incl_A), angle=PA_A,
                                fill=False, edgecolor='orange', ls='-', lw=5, label='R = 334 AU')
            ellipse_B1 = Ellipse(xy=(posx_B, posy_B),
                                 width=r_B1, height=r_B1*np.sin(incl_B), angle=PA_B,
                                 fill=False, edgecolor='orange', ls='-', lw=5, label='R = 324 AU')
            ellipse_B2 = Ellipse(xy=(posx_B, posy_B),
                                 width=r_B2, height=r_B2*np.sin(incl_B), angle=PA_B,
                                 fill=False, edgecolor='r', ls='-', lw=5, label='R = 145 AU')
            ax.add_artist(ellipse_A)
            ax.add_artist(ellipse_B1)
            ax.add_artist(ellipse_B2)

        # Annotate with some text:
        freq = str(round(lines[mol]['restfreq'], 2)) + ' GHz'
        trans = '({}-{})'.format(lines[mol]['jnum'] + 1, lines[mol]['jnum'])
        molname = r'HCO$^+$(4-3)' if mol is 'hco' else mol.upper() + trans
        sysname = 'd253-1536'

        # Print the system name.
        ax.text(1.8, 1.6, sysname,
                fontsize=20, weight='bold', horizontalalignment='left',
                path_effects=[PathEffects.withStroke(linewidth=2,
                                                     foreground="w")])

        ax.text(-1.85, 1.7, molname,
                fontsize=13, weight='bold', horizontalalignment='right',
                path_effects=[PathEffects.withStroke(linewidth=1,
                                                     foreground="w")])

        ax.text(-1.8, 1.5, freq,
                fontsize=13, horizontalalignment='right',
                path_effects=[PathEffects.withStroke(linewidth=1,
                                                     foreground="w")])

        # Add figure text
        # if text is not None:
        #     for t in text:
        #         ax.text(1, 1, *t, fontsize=18,
        #                 path_effects=[PathEffects.withStroke(linewidth=3,
        #                                                      foreground="w")])
        if self.title:
            plt.suptitle(self.title, weight='bold')


modeling = '/Volumes/disks/jonas/modeling/'
path_hco, path_hcn = modeling + 'data/hco/hco-short110.fits', modeling + 'data/hcn/hcn-short80.fits'
path_co, path_cs = modeling + 'data/co/co-short60.fits', modeling + 'data/cs/cs-short0.fits'
path_modelhco = modeling + 'gridsearch_runs/jan21_hco/jan21_hco_bestFit.fits'

model_hcn_april9 = modeling + 'mcmc_runs/april9-hcn/model_files/april9-hcn_bestFit.fits'
model_hcn_april9_resid = modeling + 'mcmc_runs/april9-hcn/model_files/april9-hcn_bestFit_resid.fits'

model_co_april9 = modeling + 'mcmc_runs/april9-co/model_files/april9-co_bestFit.fits'
model_co_april9_resid = modeling + 'mcmc_runs/april9-co/model_files/april9-co_bestFit_resid.fits'

model_hco_april9 = modeling + 'mcmc_runs/april9-hco/model_files/april9-hco_bestFit.fits'
model_hco_april9_resid = modeling + 'mcmc_runs/april9-hco/model_files/april9-hco_bestFit_resid.fits'




# f = Figure([path_hcn], moment=1, remove_bg=True, save=True)

# f2 = Figure([path_hco, path_co], moment=1)

# f_hco_a9 = Figure([path_hco, model_hco_april9, model_hco_april9_resid], moment=0, remove_bg=True, save=True)
# f_co_a9 = Figure([path_co, model_co_april9, model_co_april9_resid], moment=0, remove_bg=True, save=True)
# f_hcn_a9 = Figure([path_hcn, model_hcn_april9, model_hcn_april9_resid], moment=0, remove_bg=True, save=True)

# fig35_hco = Figure(['data/hco/hco.fits', 'data/hco/hco-short110.fits'], moment=0, remove_bg=True, save=True)
# fig36_hco = Figure(['data/hco/hco.fits', 'data/hco/hco-short110.fits'], moment=1, remove_bg=True, save=True)
#

# fig35_hcn = Figure(['data/hcn/hcn.fits', 'data/hcn/hcn-short80.fits'], moment=0, remove_bg=True, save=True)
# fig36_hcn = Figure(['data/hcn/hcn.fits', 'data/hcn/hcn-short80.fits'], moment=1, remove_bg=True, save=True)
#
# fig35_co = Figure(['data/co/co.fits', 'data/co/co-short60.fits'], moment=0, remove_bg=True, save=True)
# fig36_co = Figure(['data/co/co.fits', 'data/co/co-short60.fits'], moment=1, remove_bg=True, save=True)


# test = Figure('data/hco/hco.fits', moment=0, remove_bg=True, save=False, title='Test Title')











class GridSearch_Run:
    def __init__(self, path, save_all_plots=False):
        """
        Initialize the object.

        Args:
            path (str): Path to the run dir, plus base name for the files contained therein.
                        Ex: 'gridsearch_runs/jan10_hco/jan10_hco'
            save_all_plots (bool): If true, run all plotting functions and save
                                   the resulting plots.
        """
        self.path = path
        self.mol = self.get_line()
        print((self.mol, lines[self.mol]['baseline_cutoff']))
        self.run_date = path.split('/')[-1].split('_')[0]
        self.out_path = './gridsearch_results/{}-{}'.format(self.run_date, self.mol)
        self.data_path = './data/{}/{}-short{}.fits'.format(self.mol, self.mol,
                                                            str(lines[self.mol]['baseline_cutoff']))

        log = self.depickleLogFile()
        self.steps = log[0]
        self.raw_x2 = min(log[1][0])
        self.red_x2 = min(log[1][1])

        self.model_image = fits.getdata(self.path + '_bestFit.fits', ext=0).squeeze()
        self.model_header = fits.getheader(self.path + '_bestFit.fits', ext=0)
        self.data_image = fits.getdata(self.data_path, ext=0).squeeze()
        self.data_header = fits.getheader(self.data_path, ext=0)

        if save_all_plots:
            self.plot_all()



    def get_line(self):
        for mol in ['hco', 'hcn', 'co', 'cs']:
            if mol in self.path:
                break
        return mol


    def depickleLogFile(self):
        """
        Read in the pickle'd full-log file from a run.

        This can be cleaned up significantly, but is functional.
        """
        df = pickle.load(open(self.path + '_step-log.pickle', 'rb'))
        df_a, df_b = df.loc['A', :], df.loc['B', :]
        min_X2_a = min(df_a['Reduced Chi2'])
        min_X2_b = min(df_b['Reduced Chi2'])
        best_fit_a = df_a.loc[df_a['Reduced Chi2'] == min_X2_a]
        best_fit_b = df_b.loc[df_b['Reduced Chi2'] == min_X2_b]
        X2s = [df_a['Raw Chi2'], df_a['Reduced Chi2']]
        del df_a['Reduced Chi2']
        del df_a['Raw Chi2']
        disk_A, disk_B = [], []
        [ disk_A.append({}) for i in df_a ]
        [ disk_B.append({}) for i in df_a ]
        for i, p in enumerate(df_a):
            if p != 'Raw Chi2' and p != 'Reduced Chi2':
                ps_A = df_a[p]
                disk_A[i]['p_min'] = min(ps_A)
                disk_A[i]['p_max'] = max(ps_A)
                disk_A[i]['best_fits'] = list(best_fit_a[p])
                disk_A[i]['xvals_queried'] = list(set(ps_A))
                disk_A[i]['name'] = p
                ps_B = df_b[p]
                disk_B[i]['p_min'] = min(ps_B)
                disk_B[i]['p_max'] = max(ps_B)
                disk_B[i]['best_fits'] = list(best_fit_b[p])
                disk_B[i]['xvals_queried'] = list(set(ps_B))
                disk_B[i]['name'] = p

        both_disks = [disk_A, disk_B]
        return (both_disks, X2s)


    def plot_step_duration(self, ns=[10, 20, 50], save=False):
        """Plot how long each step took, plus some smoothing stuff.

        Args:
            ns (list of ints): A list of the smoothing windows to use.
                               Note len(ns) can't be longer than 5 without adding
                               more colors to colors list.
        """
        plt.close()
        print("\nPlotting step durations...")

        data = pd.read_csv(self.path + '_stepDurations.csv', sep=',')
        xs = data['step']
        ys = data['duration'] / 60

        def get_rolling_avg(xs, ys, n):
            # avg_ys = []
            # for i in range(n / 2, len(ys) - n / 2):
            #    avg_y = sum(ys[i - n / 2:i + n / 2]) / n
            #    avg_ys.append(avg_y)

            avg_ys = [sum(ys[i - n / 2:i + n / 2]) / n
                      for i in range(n / 2, len(ys) - n / 2)]
            return avg_ys


        plt.figure(figsize=(7, 5))
        plt.plot(xs, ys, '-k', linewidth=0.1, label='True time')
        colors = ['orange', 'red', 'blue', 'green', 'yellow']
        for i in range(len(ns)):
            n = ns[i]
            avg_ys = get_rolling_avg(xs, ys, n)
            plt.plot(xs[n / 2:-n / 2], avg_ys, linestyle='-', color=colors[i],
                     linewidth=0.1 * n, label=str(n) + '-step smoothing')

        # Not sure this is right.
        plt.legend()
        plt.xlabel('Step', fontweight='bold')
        plt.ylabel('Time (minutes)', fontweight='bold')
        plt.title('Time per Step for Grid Search Run for ' + self.run_date,
                  fontweight='bold', fontsize=14)
        if save is True:
            plt.savefig(self.out_path + '_durations.pdf')
            print(("Saved to " + self.out_path + '_durations.pdf'))
        else:
            plt.show()
        plt.clf()


    def plot_best_fit_params(self, save=False):
        """
        Plot where the best-fit values from a grid search fall.

        Plot where the best-fit value(s) stand(s) relative to the range queried in
        a given grid search run.

        Args:
            fname (str): Name of the pickled step log from the grid search.
            Assumes fname is './models/dateofrun/dateofrun'
        """
        plt.close()
        print("\nPlotting best-fit param number lines...")

        run_date = self.run_date
        # both_disks = self.steps

        # Don't plot the parameters that weren't fit.
        # Keep the statics in case we want to do something with them later
        disk_A_full, disk_B_full = self.steps
        disk_A, disk_B = [], []
        disk_A_statics, disk_B_statics = [], []
        for param in disk_A_full:
            if len(param['xvals_queried']) > 1:
                disk_A.append(param)
            else:
                disk_A_statics.append(param)

        for param in disk_B_full:
            if len(param['xvals_queried']) > 1:
                disk_B.append(param)
            else:
                disk_B_statics.append(param)
        both_disks = [disk_A, disk_B]

        colors = ['red', 'blue']
        height = max(len(disk_A), len(disk_B)) + 1

        f, axarr = plt.subplots(height, 2, figsize=[8, height+2])
        axarr[(0, 0)].axis('off')
        axarr[(0, 1)].axis('off')
        axarr[(0, 0)].text(0.2, -0.2, 'Summary of\n' + run_date + ' Run',
                           fontsize=16, fontweight='bold')
        str_rawX2 = str(round(self.raw_x2, 2))
        str_redX2 = str(round(self.red_x2, 6))
        chi_str = '       Min. Raw Chi2: {}\nMin. Reduced Chi2: {}'.format(str_rawX2, str_redX2)
        axarr[(0, 1)].text(0, 0, chi_str, fontsize=10)
        for d in [0, 1]:
            params = both_disks[d]
            for i, p in enumerate(params, 1):
                xs = np.linspace(p['p_min'], p['p_max'], 2)
                axarr[(i, d)].set_title(p['name'], fontsize=10, weight='bold')
                axarr[(i, d)].yaxis.set_ticks([])
                axarr[(i, d)].xaxis.set_ticks(p['xvals_queried'])
                if len(p['xvals_queried']) > 5:
                    axarr[(i, d)].set_xticklabels(p['xvals_queried'],
                                                  rotation=45)
                axarr[(i, d)].plot(xs, [0] * 2, '-k')
                for bf in p['best_fits']:
                    # Make the opacity proportional to how many best fits there are.
                    a = 1 / (2 * len(p['best_fits']))
                    axarr[(i, d)].plot(bf, 0, marker='o', markersize=10,
                                       color='black', alpha=a)
                    axarr[(i, d)].plot(bf, 0, marker='o', markersize=9,
                                       color=colors[d], markerfacecolor='none',
                                       markeredgewidth=3)
                # It'd be nice to not have it fill empty spaces with blank grids.
                # if len(params) < height:


        plt.tight_layout()
        if save is True:
            plt.savefig(self.out_path + '_bestfit_params.pdf')
            print(("Saved to " + self.out_path + '_bestfit_params.pdf'))
        else:
            plt.show()


    def DMR_images(self, cmap='seismic', save=False):
        """Plot a triptych of data, model, and residuals.

        It would be nice to have an option to also plot the grid search results.
        Still need to:
        - Get the beam to plot
        - Get the velocity labels in the right places

        Some nice cmaps: magma, rainbow
        """
        plt.close()
        print("\nPlotting DMR images...")

        model_path = self.path + '_bestFit.fits'
        resid_path = self.path + '_bestFit_resid.fits'
        data_path = self.data_path
        out_path = './gridsearch_results/' + self.run_date + '_DMR-images.pdf'

        real_data = fits.getdata(data_path, ext=0).squeeze()
        image_header = fits.getheader(data_path, ext=0)
        model_data = fits.getdata(model_path, ext=0).squeeze()
        model_header = fits.getheader(model_path, ext=0)
        resid_data = fits.getdata(resid_path, ext=0).squeeze()
        resid_header = fits.getheader(resid_path, ext=0)

        # Define some plotting params
        hspace = -0.2
        wspace = -0.1

        # Set up some physical params
        vmin, vmax = np.nanmin(real_data), np.nanmax(real_data)
        offsets_dA, offsets_dB = offsets[0], offsets[1]
        x_center = int(np.floor(real_data.shape[1] / 2))
        crop_arcsec = 2
        crop_pix = int(crop_arcsec / 0.045)
        xmin, xmax = x_center - crop_pix, x_center + crop_pix
        offsets_dA_pix = [44 - offsets_dA[0]/0.045,
                          44 - offsets_dA[1]/0.045]
        offsets_dB_pix = [44 - offsets_dB[0]/0.045,
                          44 + offsets_dB[1]/0.045]

        chanstep_vel = image_header['CDELT3'] * 0.001
        chan0_vel = image_header['CRVAL3'] * 0.001 - image_header['CRPIX3'] * chanstep_vel
        chan_offset = 15
        nchans = 30

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
        n_rows = int(np.floor(np.sqrt(nchans))) + 1
        n_cols = int(np.ceil(np.sqrt(nchans)))

        # Get the plots going
        # fig = plt.figure(figsize=(n_rows * 3, 7))
        fig = plt.figure(figsize=(18, n_rows + 1))
        big_fig = gridspec.GridSpec(1, 3)

        data_ims = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                                                    subplot_spec=big_fig[0],
                                                    wspace=wspace, hspace=hspace)
        model_ims = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                                                     subplot_spec=big_fig[1],
                                                     wspace=wspace, hspace=hspace)
        resid_ims = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                                                     subplot_spec=big_fig[2],
                                                     wspace=wspace, hspace=hspace)
        # Populate the plots
        print("Got the necessary info; now plotting...")
        for i in range(nchans):
            chan = i + chan_offset
            velocity = str(round(chan0_vel + chan * chanstep_vel, 2))
            ax_d = plt.Subplot(fig, data_ims[i])
            ax_m = plt.Subplot(fig, model_ims[i])
            ax_r = plt.Subplot(fig, resid_ims[i])

            if i == int(np.floor(n_rows / 2)):
                ax_d.set_title('Data', weight='bold')
                ax_m.set_title('Model', weight='bold')
                ax_r.set_title('Residuals', weight='bold')

            # Plot the data
            im_d = ax_d.imshow(real_data[i + chan_offset][xmin:xmax, xmin:xmax],
                               cmap=cmap, vmin=vmin, vmax=vmax)
            im_m = ax_m.imshow(model_data[i + chan_offset][xmin:xmax, xmin:xmax],
                               cmap=cmap, vmin=vmin, vmax=vmax)
            im_r = ax_r.imshow(resid_data[i + chan_offset][xmin:xmax, xmin:xmax],
                               cmap=cmap, vmin=vmin, vmax=vmax)


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
            ax_d.text(44, 80, velocity + ' km/s', fontsize=6, color='w',
                    horizontalalignment='center', verticalalignment='center')
            ax_m.text(44, 80, velocity + ' km/s', fontsize=6, color='w',
                    horizontalalignment='center', verticalalignment='center')
            ax_r.text(44, 80, velocity + ' km/s', fontsize=6, color='w',
                    horizontalalignment='center', verticalalignment='center')

            if i == n_rows * (n_cols - 2) and add_beam_d is True:
                el = Ellipse(xy=[0.8 * crop_arcsec, 0.8 * crop_pix],
                             width=bmin, height=bmaj, angle=-bpa,
                             fc='k', ec='w', fill=False, hatch='////////')
                ax_d.add_artist(el)

            fig.add_subplot(ax_m)
            fig.add_subplot(ax_d)
            fig.add_subplot(ax_r)
            fig.tight_layout()


        cmaps = imshow(real_data[i + chan_offset][xmin:xmax, xmin:xmax],
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=(crop_arcsec, -crop_arcsec, crop_arcsec, -crop_arcsec))

        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        cax = plt.axes([0.2, 0.06, 0.6, 0.07])
        cbar = colorbar(cmaps, cax=cax, orientation='horizontal')
        cbar.set_label('Jy/beam', labelpad=-12, fontsize=12, weight='bold')
        cbar.set_ticks([vmin, vmax])


        fig.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=0.0, top=0.93)

        # No idea why this is here
        plt.close()

        if save is True:
            fig.savefig(out_path)
            print(("Saved to " + out_path))
        else:
            print("Showing")
            fig.show()


    def DMR_spectra(self, save=False):
        """
        Plot a model/data/resid triptych of spectra
        y-axis units: each pixel is in Jy/beam, so want to:
            - Multiply each by beam
            - Divide by number of pix (x*y)?
        """
        plt.close()
        print("\nPlotting DMR spectra...")

        model_spec = np.array([np.sum(self.model_image[i])/self.model_image.shape[1]
                               for i in range(self.model_image.shape[0])])
        data_spec = np.array([np.sum(self.data_image[i])/self.data_image.shape[1]
                              for i in range(self.data_image.shape[0])])
        resid_spec = data_spec - model_spec

        chans = np.arange(len(model_spec))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.plot(data_spec, color='steelblue')
        ax2.plot(model_spec, color='steelblue')
        ax3.plot(resid_spec, color='steelblue')

        ax1.set_title('Data', weight='bold')
        ax2.set_title('Model', weight='bold')
        ax3.set_title('Residuals', weight='bold')
        ax1.grid(False), ax2.grid(False), ax3.grid(False)

        ymin = min([min(l) for l in [model_spec, data_spec, resid_spec]])
        ymax = max([max(l) for l in [model_spec, data_spec, resid_spec]])
        ax1.set_xlabel('Channel'), ax1.set_ylabel('Jy/Beam')
        ax1.set_ylim(ymin, ymax), ax2.set_ylim(ymin, ymax), ax3.set_ylim(ymin, ymax)
        plt.tight_layout()
        sns.despine()

        if save:

            fig(self.out_path + '_DMR-spectra.pdf')
            print(("Saved to " + self.out_path + '_DMR-spectra.pdf'))
        else:
            plt.show()


    def param_degeneracies(self, DI=0, save=False):
        """
        Plot Chi2 as a function of two params.

        I think this works now.
        """

        df_raw = pickle.load(open(('{}_step-log.pickle').format(self.path), 'rb'))
        df_full = df_raw.loc['A', :] if DI == 0 else df_raw.loc['B', :]
        l = list(df_full.columns)

        for i, p in enumerate(l):
            len_p = len(list(set(df_full[p])))
            if len_p > 1:
                if p != 'Reduced Chi2' and p != 'Raw Chi2':
                    print((str([i]), p, '(Length: ' + str(len_p) + ')'))

        p1_idx = int(eval(input('Select the index of the first parameter.\n')))
        p2_idx = int(eval(input('Select the index of the second parameter.\n')))
        param1, param2 = l[p1_idx], l[p2_idx]

        # Clear out the parameters that we're not interested in
        l.remove(param1)
        l.remove(param2)
        l.remove('Raw Chi2')
        df = df_full
        for p in l:
            p_vals = list(set(df[p]))
            if len(p_vals) == 1 or p == 'Reduced Chi2':
                df = df.drop(p, axis=1)
            else:
                df = df.drop(df[df[p] != df[p][0]].index)
        df = df.reset_index(drop=True)

        # Make sure we're looking at an iterated parameter.
        len_p1, len_p2 = len(list(set(df[param1]))), len(list(set(df[param2])))
        print((len_p1, len_p2))
        if len_p1 < 2 or len_p2 < 2:
            return 'Use parameters of length greater than 1'

        """
        I think this was trying to force a landscape layout.
        Could consider using mat.T instead?
        if df[param1][0] == df[param1][1]:
            p_i, p_j = param2, param1
            len_p_i, len_p_j = len_p2, len_p1
        else:
            p_i, p_j = param1, param2
            len_p_i, len_p_j = len_p1, len_p2
        """
        p_i, p_j = param1, param2
        len_p_i, len_p_j = len_p1, len_p2

        # Populate our chi-squared grid
        mat = np.zeros((len_p_i, len_p_j))
        for i in range(len_p_i):
            for j in range(len_p_j):
                this_chi = df['Raw Chi2'][i * len_p_j + j]
                mat[(i, j)] = this_chi
                # print this_chi
                # print p_i, df[p_i][i * len_p_j + j], '; ', p_j, df[p_j][i * len_p_j + j]
                # print

        plt.close()
        # vmin, vmax = np.nanmin(df_full['Raw Chi2']), np.nanmax(df_full['Raw Chi2'])
        plt.matshow(mat, cmap='jet')  #, vmin=vmin, vmax=vmax)
        plt.xlabel(df[p_j].name)
        plt.ylabel(df[p_i].name)
        plt.xticks(list(range(len_p2)), sorted(list(set(df[param2]))))
        plt.yticks(list(range(len_p1)), sorted(list(set(df[param1]))))
        plt.title('Chi2 Map Over Params')
        plt.grid(False, color='k')  #, alpha=0.5)
        plt.colorbar()
        plt.gca().xaxis.tick_bottom()

        if save is True:
            out = "{}_param_degens_disk{}.pdf".format(self.out_path, DI)
            plt.savefig(out)
            print(("Saved to ", out))
        else:
            plt.show() #block=False)
        return None


    def plot_all(self):
        self.plot_best_fit_params(save=True)
        self.plot_step_duration(save=True)
        self.DMR_spectra(save=True)
        self.DMR_images(save=True)

# run = GridSearch_Run('gridsearch_runs/jan21_hco/jan21_hco')
# run.plot_best_fit_params()





# The End
