#!/usr/bin/python


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from astropy.io import fits as pf
from matplotlib.ticker import *
import numpy as np
import matplotlib.font_manager
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.patches import Ellipse as ellipse
import seaborn as sns


# FITS Header Keywords:
# http://jsoc.stanford.edu/doc/keywords/STEREO/STEREO_site_standard_fits_keywords.txt

# Constants and conversion factors
# ---------------------------------

# Solar mass (g)
msol = 1.989e+33
# Gravitational constant (cm3 g-1 cm-2)
grav = 6.67259e-8
# Conversion factor: AU to cm
AU2cm = 1.49597871e+13
# Conversion factor: cm to km
cm2km = 1.0e-5
# Speed of light (km)
ckm = 2.99792458e+05

# Disk and star inputs
# ---------------------------------

# Distance to source (parsec)
dist = 414.
# Disk inclination (degrees --> radians)
inc = 70.*np.pi/180.0
# Position angle (degrees --> radians)
dpa = (70)*np.pi/180.0
# Radius r0 (AU) for flaring relation: z = (h0/r0)**psi
r0 = 1.0
# Scale height at r0
h0 = 0
# Flaring index
psi = 1.00
# Mass of central star (g)
mstar = 3.5*msol
# Source ocity (km/s)
source = 10.55


# To change line:
# change fits_chan index
# change color Scale (~ line 250)
# change number of panels

# Name of FITS image containing the channel maps
chanmaps = ['../chibyeye/modelA_convolved_wrestfreq.fits', '../v2434/hco.fits',
            '../v2434/hcn.fits', '../v2434/co.fits', '../v2434/cs.fits']

# Control what you look at with this index
fits_chan = chanmaps[1]


# Set the output filename
figFileName = '928_hcoData.png'

# Read the header from the model channel maps
head = pf.getheader(fits_chan)


# Generate x and y axes: offset position in arcsec
nx = head['NAXIS1']
xpix = head['CRPIX1']
xdelt = head['CDELT1']

ny = head['NAXIS2']
ypix = head['CRPIX2']
ydelt = head['CDELT2']

# Convert from degrees --> arsec --> AU --> cm
xorig = ((arange(nx) - xpix + 1) * xdelt)*3600
yorig = ((arange(ny) - ypix + 1) * ydelt)*3600

# Source is contained within 4" x 4" - clip the x and y axes extent
xval = xorig[(xorig >= -4.0) & (xorig <= 4.0)]*dist*AU2cm
yval = yorig[(yorig >= -4.0) & (yorig <= 4.0)]*dist*AU2cm

# Make 2D array containing all (x,y) axis coordinates
xdisk, ydisk = np.meshgrid(xval, yval)


# Read the header from the channel map
head = pf.getheader(fits_chan)

# Create channel axis (km/s)
nv = head['NAXIS3']								# number of channels
vpix = head['CRPIX3']								# val is 1
vdelt = head['CDELT3']								# val is 4e02
vval = head['CRVAL3']								# val is 0
vel = ((arange(nv) - vpix + 1) * vdelt) + vval

# Extract rest frequency (Hz)
freq = head['RESTFRQ']

# Convert from frequency (Hz) to LRSK velocity (km/s) using
# rest frequency and source velocity
for i in range(nv):
    vel[i] = (ckm*((freq-vel[i])/freq))-source


# -----------------------------------------------------------------------
# Plot channel maps and isovelocity contours
# -----------------------------------------------------------------------

# Use original x and y axes coordinates for the channel images
img_chan = {'image': squeeze(pf.getdata(fits_chan)),
            'ra_offset': xorig, 'dec_offset': yorig}

# Set spacing between axes labels and tick direction
rcParams['ytick.major.pad'] = 6
rcParams['xtick.major.pad'] = 6

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

rcParams['xtick.major.size'] = 5
rcParams['xtick.minor.size'] = 2.5
rcParams['ytick.major.size'] = 5
rcParams['ytick.minor.size'] = 2.5

rcParams['ytick.labelsize'] = 12
rcParams['xtick.labelsize'] = 12

# Set seaborn plot styles
sns.set_style("white")
# sns.set_style("ticks")

# Set figure size and create image canvas (in cm)
fig = figure(figsize=[7, 7])

# Set axes limits
xmin = -2.0
xmax = 2.0
ymin = -2.0
ymax = 2.0

# Set physical range of colour map
cxmin = img_chan['ra_offset'][0]
cxmax = img_chan['ra_offset'][-1]
cymin = img_chan['dec_offset'][-1]
cymax = img_chan['dec_offset'][0]

# Set limits and tics of colorbar
cbmin = -0.0
cbmax = 1							# HCO: 1, HCN: 0.25, CO: 1, CS: 0.05
cbtmaj = 0.5
cbtmin = 0.05
cbnum = int((cbmax-cbmin)/cbtmin)

# Set colorpalette
#cpal = colors.ListedColormap(sns.cubehelix_palette(cbnum,start=0.5,rot=-0.8,light=0.05,dark=0.95,hue=0.75,reverse=True,gamma=1.0))
cpal = cm.afmhot
#cpal = cm.gist_heat


# Adjust spacing between subplots
subplots_adjust(wspace=0.1, hspace=0.1)


# Limit isovelocity contour extent: within [0:67]
c1 = 12
c2 = 55

bmin = head['bmin'] * 3600.
bmaj = head['bmaj'] * 3600.
bpa = head['bpa']

# Loop over channels and plot each panel
for i in range(20):
    # for hco: 20 steps (5,4)
    # for hcn: 25 steps (5,5)

    chan = 5+i 					# for model
    # chan = 15+i				# for hco
    # chan = 13+i				# for hcn
    # chan = 35+i				# for co
    # chan = 35+i				# for cs
    velocity = '%4.2f' % (vel[chan])

    ax = fig.add_subplot(4, 5, i+1)
    ax.set_xlim(xmax, xmin)
    ax.set_ylim(ymin, ymax)
    ax.grid(False)

    majorLocator = MultipleLocator(2)
    ax.xaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_locator(majorLocator)

    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Only plot the beam in the lower left corner
    if i == 15:
        el = ellipse(xy=[3, -3], width=bmin, height=bmaj,
                     angle=-bpa, fc='k', ec='w', fill=True, zorder=10)
        ax.add_artist(el)

    text(1.7, 1.5, velocity+' km/s', fontsize=8, color='w')

    cmap = imshow(img_chan['image'][:][:][chan],
                  extent=[cxmin, cxmax, cymin, cymax],
                  vmin=cbmin,
                  vmax=cbmax,
                  interpolation='bilinear',
                  cmap=cpal)


# Create and add a colour bar with label and tics
# Physical coords:
cax = fig.add_axes([0.92, 0.12, 0.02, 0.75])

cbar = colorbar(cmap, cax=cax)
cbar.set_label('Jy/beam', labelpad=-10, fontsize=12)
# cbar.set_ticks(np.arange(0,cbmax,cbtmaj))
minorLocator = LinearLocator(cbnum+1)
cbar.ax.yaxis.set_minor_locator(minorLocator)
cbar.update_ticks()

# Save the figure to pdf/eps

fig.savefig(figFileName)
plt.show()
