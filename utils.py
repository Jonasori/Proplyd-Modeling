"""Some nice functions to be used in making models."""


############
# PACKAGES #
############

import numpy as np
import subprocess as sp
from astropy.io import fits
from disk_model.disk import Disk
import disk_model.raytrace as rt

# Local package files
from constants import obs_stuff, other_params, get_data_path, lines #, mol

######################
# CONSTANTS & PARAMS #
######################


# Some other constants
# col_dens, Tfo, m_star, r_in, rotHand, offsets, distance = other_params




####################
# USEFUL FUNCTIONS #
####################


def makeModel(diskParams, outputPath, DI, mol, short_vis_only=True):
    """Make a single model disk.

    Args:
        diskParams (dict of floats): the physical parameters for the model.
        outputPath (str): The path to where created files should go,
                          including filename.
        DI (0 or 1): the index of the disk being modeled.
    Creates:
        {outputPath}.fits, a single model disk.
    """
    # DI = Disk Index: the index for the tuples below. 0=A, 1=B

    print "[ Entering makeModel() ]"

    # Get line-specific stuff
    vsys, restfreq, freq0, obsv, chanstep, n_chans, chanmins, jnum = obs_stuff(mol, short_vis_only=short_vis_only)

    # Clear out space
    # sp.call('rm -rf {}.{{fits,vis,uvf,im}}'.format(outputPath), shell=True)

    v_turb   = diskParams['v_turb']
    zq       = diskParams['zq']
    r_crit   = diskParams['r_crit']
    rho_p    = diskParams['rho_p']
    t_mid    = diskParams['t_mid']
    PA       = diskParams['PA']
    incl     = diskParams['incl']
    pos_x    = diskParams['pos_x']
    pos_y    = diskParams['pos_y']
    v_sys    = diskParams['v_sys']
    t_atms   = diskParams['t_atms']
    t_qq     = diskParams['t_qq']
    r_out    = diskParams['r_out']
    m_disk   = diskParams['m_disk']
    x_mol    = diskParams['x_mol']

    col_dens = lines[mol]['col_dens']
    t_fo     = lines[mol]['t_fo']
    m_star   = lines[mol]['m_star']
    r_in     = lines[mol]['r_in']
    rotHand  = lines[mol]['rotHand']
    offsets  = lines[mol]['offsets']
    distance = lines[mol]['distance']


    a = Disk(params=[t_qq,
                     10**m_disk,
                     rho_p,
                     r_in[DI],
                     r_out,
                     r_crit,
                     incl,
                     m_star[DI],
                     10**x_mol,
                     v_turb,
                     zq,
                     t_mid,
                     t_atms,
                     col_dens,
                     [1., r_out],
                     rotHand[DI]])

    # The data have 51 channels (from the casa split()), so n_chans must be 51
    rt.total_model(a,
                   imres=0.045,
                   nchans=n_chans[DI],
                   chanmin=chanmins[DI],
                   chanstep=chanstep,
                   distance=389,
                   xnpix=256,
                   vsys=v_sys,
                   PA=PA,
                   offs=[pos_x, pos_y],
                   modfile=outputPath,
                   isgas=True,
                   flipme=False,
                   freq0=restfreq,
                   Jnum=jnum,
                   obsv=obsv)

    print "MakeModel() completed"


# SUM TWO MODEL DISKS #
def sumDisks(filePathA, filePathB, outputPath, mol):
    """Sum two model disks.

    Args:
        filePathA, filePathB (str): where to find the model files.
                                    Don't include the filetype extension.
        outputPath: the name of the file to be exported.
                    Don't include the filetype extension.
    Creates:	outputPath.[fits, im, vis, uvf]
    """

    dataPath = get_data_path(mol)
    vsys, restfreq, freq0, obsv, chanstep, n_chans, chanmins, jnum = obs_stuff(mol, short_vis_only=True)

    # Now sum them up and make a new fits file
    a = fits.getdata(filePathA + '.fits')
    b = fits.getdata(filePathB + '.fits')

    # The actual disk summing
    sum_data = a + b

    # There are too many variable names here and they're confusing.

    # Create the empty structure for the final fits file and add the data
    im = fits.PrimaryHDU()
    im.data = sum_data

    # Add the header. Kevin's code should populate the header more or less
    # correctly, so pull a header from one of the models.
    with fits.open(filePathA + '.fits') as model:
        model_header = model[0].header
    im.header = model_header

    # Now swap out some of the values using values from the data file:
    with fits.open(dataPath + '.fits') as data:
        data_header = data[0].header

    # Does the fact that I have to change these reflect a deeper problem?
    # They are RA and DEC, and are both 0.0 straight out of the model.
    im.header['CRVAL1'] = data_header['CRVAL1']
    im.header['CRVAL2'] = data_header['CRVAL2']
    """
    im.header['CDELT1'] = data_header['CDELT1']
    im.header['CDELT2'] = data_header['CDELT2']
    """
    im.header['RESTFREQ'] = data_header['RESTFREQ']
    # Ok to do this since velocity axis is labeled VELO-LSR and
    # but the header doesn't have a SPECSYS value yet.
    # im.header['SPECSYS'] = 'LSRK'
    # im.header['EPOCH'] = data_header['EPOCH']

    im.writeto(outputPath + '.fits', overwrite=True)

    # Clear out the old files to make room for the new
    sp.call('rm -rf {}.im'.format(outputPath), shell=True)
    sp.call('rm -rf {}.uvf'.format(outputPath), shell=True)
    sp.call('rm -rf {}.vis'.format(outputPath), shell=True)
    print "Deleted .im, .uvf, and .vis\n"


def chiSq(infile, mol, cut_central_chans=False):
    """Calculate chi-squared metric between model and data.

    Args:
        infile: file name of model to be compared, not including .fits
    Returns:	[Raw X2, Reduced X2]
    Creates: 	None
    """

    dataPath = get_data_path(mol)
    # GET VISIBILITIES
    with fits.open(dataPath + '.uvf') as data:
        data_vis = data[0].data['data'].squeeze()

    with fits.open(infile + '.uvf') as model:
        model_vis = model[0].data['data'].squeeze()

    # PREPARE STUFF FOR CHI SQUARED

    # Turn polarized data to stokes
    # data_vis.squeeze(): [visibilities, chans, polarizations?, real/imaginary]
    # Should the 2s be floats?
    if cut_central_chans is False:
        data_real = (data_vis[:, :, 0, 0] + data_vis[:, :, 1, 0])/2.
        data_imag = (data_vis[:, :, 0, 1] + data_vis[:, :, 1, 1])/2.

        # Get real and imaginary vals, skipping repeating vals created by uvmodel.
        # When uvmodel converts to Stokes I, it either puts the value in place
        # of BOTH xx and yy, or makes xx and yy the same.
        # Either way, selecting every other value solves the problem.
        model_real = model_vis[::2, :, 0]
        model_imag = model_vis[::2, :, 1]

        wt = data_vis[:, :, 0, 2]
        raw_chi = np.sum(wt*(data_real - model_real)**2) + \
            np.sum(wt*(data_imag - model_imag)**2)


    elif cut_central_chans is True:
        # Define the bounds of the slice that's getting pulled out
        slice_front = 26
        slice_back = 34

        data_real_front = (data_vis[:, :slice_front, 0, 0] + data_vis[:, :slice_front, 1, 0])/2.
        data_real_back = (data_vis[:, slice_back:, 0, 0] + data_vis[:, slice_back:, 1, 0])/2.

        data_imag_front = (data_vis[:, :slice_front, 0, 1] + data_vis[:, :slice_front, 1, 1])/2.
        data_imag_back = (data_vis[:, slice_back:, 0, 1] + data_vis[:, slice_back:, 1, 1])/2.

        model_real_front = model_vis[::2, :slice_front, 0]
        model_real_back = model_vis[::2, slice_back:, 0]

        model_imag_front = model_vis[::2, :slice_front, 1]
        model_imag_back = model_vis[::2, slice_back:, 1]

        wt_front = data_vis[:, :slice_front, 0, 2]
        wt_back = data_vis[:, slice_back:, 0, 2]

        # Do chi-front, chi-back and then just sum them instead of cat'ing

        raw_chi_front = np.sum(wt_front * (data_real_front - model_real_front)**2) + \
            np.sum(wt_front*(data_imag_front - model_imag_front)**2)

        raw_chi_back = np.sum(wt_back * (data_real_back - model_real_back)**2) + \
            np.sum(wt_back * (data_imag_back - model_imag_back)**2)

        raw_chi = raw_chi_back + raw_chi_front

    # Degrees of freedom is how many total real and imaginary weights exist.
    dof = 2 * len(data_vis)
    red_chi = raw_chi/dof
    return [raw_chi, red_chi]
