"""Some nice functions to be used in making models."""


############
# PACKAGES #
############
import os, yaml
import numpy as np, subprocess as sp, matplotlib.pyplot as plt
from yaml import CLoader, CDumper
from astropy.io import fits

from disk_model3.disk import Disk
import disk_model3.raytrace as rt

from constants import obs_stuff, get_data_path, lines, mol
from tools import remove



# MCMC Model Class
class Model:
    """A nice way to wrap up all the model creation steps into one."""

    def __init__(self, mol, param_path, run_path, model_name, set_face_on=False):
        """
        Feed the model what it needs.

        Args:
            run_path: path to the directory where the relevant stuff is.
            model_name: run_name + unique ID number
                        (to avoid crashes between models)
        """
        self.mol = mol
        self.param_dict = yaml.load(open('{}'.format(param_path),
                                         'r'), Loader=CLoader)

        if self.param_dict['cut_baselines'] is True:
            fname_end = '-short' + str(self.param_dict['baseline_cutoff'])
        else:
            fname_end = ''
        self.datafiles_path = './data/{}/{}'.format(mol, mol + fname_end)
        self.modelfiles_path = '{}model_files/{}'.format(run_path,
                                                         model_name)
        # Not sure why these were lists. Changing to just use singulars.
        self.raw_chis, self.reduced_chis = [], []
        self.raw_chi, self.reduced_chi = np.inf, np.inf
        # This is basically just for
        # analysis.MCMC_Analysis.get_disk_objects(), which in turn is
        # just for MCMC_Analysis.plot_structure(). Should be a rare use.
        if set_face_on:
            self.param_dict['incl_A'] = 0
            self.param_dict['incl_B'] = 0

        # self.delete()  # delete any preexisting files that will conflict

    def delete(self):
        """Delete anything with this path."""
        sp.call('rm -rf {}*'.format(self.modelfiles_path), shell=True)

    def make_fits(self):
        """
        Make a disk_model from a param list.

        Output:
            (model.path).fits
        """
        # Make Disk 1
        DI = 0
        d1 = Disk(params=[self.param_dict['temp_struct_A'],
                          10**self.param_dict['m_disk_A'],
                          self.param_dict['surf_dens_str_A'],
                          self.param_dict['r_ins'][DI],
                          self.param_dict['r_out_A'],
                          self.param_dict['r_crit'],
                          self.param_dict['incl_A'],
                          self.param_dict['m_stars'][DI],
                          10**self.param_dict['mol_abundance_A'],
                          self.param_dict['v_turb'],
                          self.param_dict['vert_temp_str'],
                          self.param_dict['T_mids'][DI],
                          self.param_dict['atms_temp_A'],
                          self.param_dict['column_densities'],
                          [self.param_dict['r_ins'][DI], self.param_dict['r_out_A']],
                          self.param_dict['rot_hands'][DI]],
                  rtg=False)
        d1.Tco = self.param_dict['T_freezeout']
        d1.set_rt_grid()
        rt.total_model(d1,
                       imres=self.param_dict['imres'],
                       distance=self.param_dict['distance'],
                       chanmin=self.param_dict['chanmins'][DI],
                       nchans=self.param_dict['nchans'][DI],
                       chanstep=self.param_dict['chanstep'],
                       flipme=False,
                       Jnum=self.param_dict['jnum'],
                       freq0=self.param_dict['restfreq'],
                       xnpix=self.param_dict['imwidth'],
                       vsys=self.param_dict['vsys'][DI],
                       PA=self.param_dict['pos_angle_A'],
                       offs=self.param_dict['offsets'][DI],
                       modfile=self.modelfiles_path + '-d1',
                       obsv=self.param_dict['obsv'],
                       isgas=True,
                       hanning=True
                       )


        # Now do Disk 2
        DI = 1
        d2 = Disk(params=[self.param_dict['temp_struct_B'],
                          10**self.param_dict['m_disk_B'],
                          self.param_dict['surf_dens_str_B'],
                          self.param_dict['r_ins'][DI],
                          self.param_dict['r_out_B'],
                          self.param_dict['r_crit'],
                          self.param_dict['incl_B'],
                          self.param_dict['m_stars'][DI],
                          10**self.param_dict['mol_abundance_B'],
                          self.param_dict['v_turb'],
                          self.param_dict['vert_temp_str'],
                          self.param_dict['T_mids'][DI],
                          self.param_dict['atms_temp_B'],
                          self.param_dict['column_densities'],
                          [self.param_dict['r_ins'][DI], self.param_dict['r_out_B']],
                          self.param_dict['rot_hands'][DI]],
                  rtg=False)
        d2.Tco = self.param_dict['T_freezeout']
        d2.set_rt_grid()
        rt.total_model(d2,
                       imres=self.param_dict['imres'],
                       distance=self.param_dict['distance'],
                       chanmin=self.param_dict['chanmins'][DI],
                       nchans=self.param_dict['nchans'][DI],
                       chanstep=self.param_dict['chanstep'],
                       flipme=False,
                       Jnum=self.param_dict['jnum'],
                       freq0=self.param_dict['restfreq'],
                       xnpix=self.param_dict['imwidth'],
                       vsys=self.param_dict['vsys'][DI],
                       PA=self.param_dict['pos_angle_B'],
                       offs=self.param_dict['offsets'][DI],
                       modfile=self.modelfiles_path + '-d2',
                       obsv=self.param_dict['obsv'],
                       isgas=True,
                       hanning=True
                       )

        # Now sum those two models, make a header, and crank out some other files.
        a = fits.getdata(self.modelfiles_path + '-d1.fits')
        b = fits.getdata(self.modelfiles_path + '-d2.fits')

        # Create the empty structure for the final fits file and insert the data.
        im = fits.PrimaryHDU()
        # The actual disk summing
        im.data = a + b

        # Add the header by modifying a model header.
        with fits.open(self.modelfiles_path + '-d1.fits') as model_fits:
            model_header = model_fits[0].header
        im.header = model_header

        # Swap out some of the vals using values from the data file used by model:
        header_info_from_data = fits.open(self.datafiles_path + '.fits')
        data_header = header_info_from_data[0].header
        header_info_from_data.close()

        # Put in RA, Dec and restfreq
        im.header['CRVAL1'] = data_header['CRVAL1']
        im.header['CRVAL2'] = data_header['CRVAL2']
        im.header['RESTFRQ'] = data_header['RESTFREQ']
        im.header['SPECLINE'] = self.mol

        # Write it out to a file, overwriting the existing one if need be
        im.writeto(self.modelfiles_path + '.fits', overwrite=True)

        remove([self.modelfiles_path + '-d1.fits',
                self.modelfiles_path + '-d2.fits'])

    def obs_sample(self):
        """Create a model fits file from the data.

        Makes a model fits file with correct header information
        Samples using ALMA observation uv coverage

        Args:
            obs: an Observation instance to sample from.
        Returns:
            path.[vis, .uvf]: model files in the visibility domain.
            path.im: a model image that's just a stepping stone.
        """

        # define observation-specific model name, delete any preexisting models
        sp.call('rm -rf {}{{.im,.vis,.uvf}}'.format(self.modelfiles_path),
                shell=True)

        # Convert model into MIRIAD .im image file
        sp.call(['fits',
                 'op=xyin',
                 'in={}.fits'.format(self.modelfiles_path),
                 'out={}.im'.format(self.modelfiles_path)],
                stdout=open(os.devnull, 'wb'),
                stderr=open(os.devnull, 'wb')
                )

        # Sample the model image using the observation uv coverage
        sp.call(['uvmodel',
                 'options=replace',
                 'vis={}.vis'.format(self.datafiles_path),
                 'model={}.im'.format(self.modelfiles_path),
                 'out={}.vis'.format(self.modelfiles_path)],
                stdout=open(os.devnull, 'wb')
                )

        # Convert to UVfits
        sp.call(['fits',
                 'op=uvout',
                 'in={}.vis'.format(self.modelfiles_path),
                 'out={}.uvf'.format(self.modelfiles_path)],
                stdout=open(os.devnull, 'wb'),
                stderr=open(os.devnull, 'wb')
                )

    def chiSq(self, mol):
        """Calculate the goodness of fit between data and model."""
        # GET VISIBILITIES
        # Gotta get the Observation garbage out of here.
        data_uvf = fits.open(self.datafiles_path + '.uvf')
        data_vis = data_uvf[0].data['data'].squeeze()

        model = fits.open(self.modelfiles_path + '.uvf')
        model_vis = model[0].data['data'].squeeze()

        # PREPARE STUFF FOR CHI SQUARED

        # get real and imaginary values, skipping repeating values created by
        # uvmodel when uvmodel converts to Stokes I, it either puts the value
        # in place of BOTH xx and yy, or makes xx and yy the same.
        # Either way, selecting every other value solves the problem.

        # Turn polarized data to stokes
        # In Cail's code, this is equivalent to:
        # if data_vis.shape[2] == 2 (want [2] since we have channels in [1])
        data_real = (data_vis[:, :, 0, 0] + data_vis[:, :, 1, 0])/2.
        data_imag = (data_vis[:, :, 0, 1] + data_vis[:, :, 1, 1])/2.

        # Scout/cluster and iorek handle uvmodel differently. Read about it in S4.2 of:
        # https://github.com/kevin-flaherty/Wesleyan_cluster/blob/master/cluster_guide.pdf
        if len(model_vis.shape) == 3:
            # On iorek, the output of uvmodel is of shape
            # (2N_baselines, nchans, 3)
            model_real = model_vis[::2, :, 0]
            model_imag = model_vis[::2, :, 1]
        else:
            # If on cluster/scout, the output of uvmodel is of shape
            # (N_baselines, nchans, 2, 3), where the 2 is X or Y.
            # Since they're the same, choose either one.
            model_real = model_vis[:, :, 0, 0]
            model_imag = model_vis[:, :, 0, 1]

        wt = data_vis[:, :, 0, 2]

        # Don't fit for central channels in CO.
        if mol is not 'co':
            raw_chi = np.sum(wt * (data_real - model_real)**2 +
                             wt * (data_imag - model_imag)**2)

        else:
            # Define the bounds of the slice that's getting pulled out.
            # Found this by looking at data channel maps, identifying bad
            # central channels and their velocities, then imstating that
            # file to find their channel numbers.
            slice_front, slice_back = lines[mol]['chan_cut_idxs']

            data_real_front = (data_vis[:, :slice_front, 0, 0] + data_vis[:, :slice_front, 1, 0])/2.
            data_real_back = (data_vis[:, slice_back:, 0, 0] + data_vis[:, slice_back:, 1, 0])/2.
            data_imag_front = (data_vis[:, :slice_front, 0, 1] + data_vis[:, :slice_front, 1, 1])/2.
            data_imag_back = (data_vis[:, slice_back:, 0, 1] + data_vis[:, slice_back:, 1, 1])/2.

            if len(model_vis.shape) == 3:
                model_real_front = model_vis[::2, :slice_front, 0]
                model_real_back = model_vis[::2, slice_back:, 0]
                model_imag_front = model_vis[::2, :slice_front, 1]
                model_imag_back = model_vis[::2, slice_back:, 1]

            else:
                model_real_front = model_vis[:, :slice_front, 0, 0]
                model_real_back = model_vis[:, slice_back:, 0, 0]
                model_imag_front = model_vis[:, :slice_front, 0, 1]
                model_imag_back = model_vis[:, slice_back:, 0, 1]

            wt_front = data_vis[:, :slice_front, 0, 2]
            wt_back = data_vis[:, slice_back:, 0, 2]
            # Do chi-front, chi-back and then just sum them instead of cat'ing
            raw_chi_front = np.sum(wt_front * (data_real_front - model_real_front)**2) + \
                np.sum(wt_front*(data_imag_front - model_imag_front)**2)

            raw_chi_back = np.sum(wt_back * (data_real_back - model_real_back)**2) + \
                np.sum(wt_back * (data_imag_back - model_imag_back)**2)

            raw_chi = raw_chi_back + raw_chi_front

        # Degrees of freedom: how many total real and imaginary weights we have
        dof = 2 * len(data_vis)
        reduced_chi = raw_chi/dof

        # Maybe add the gaussian priors here?

        self.raw_chi = raw_chi
        self.reduced_chi = reduced_chi
        print("Raw Chi2: ", self.raw_chi)
        return self.raw_chi





# Gridsearch Functions
# These are probably all broken due to the deprecating of constants.py
# I never use grid search anymore, so not fixing it.

# mol = 'hco'
short_vis_only = True

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

    print("[ Entering makeModel() ]")

    # Get line-specific stuff
    vsys, restfreq, freq0, obsv, chanstep, n_chans, chanmins, jnum = obs_stuff(mol, short_vis_only=short_vis_only)


    # Clear out space
    # sp.call('rm -rf {}.{{fits,vis,uvf,im}}'.format(outputPath), shell=True)

    # Since make_diskX_params() generates
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

    # for p in diskParams.keys():
    #     print p, diskParams[p]

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
                     rotHand[DI]],
             rtg=False)
    a.Tco = param_dict['T_freezeout']
    a.set_rt_grid()
    # The data have 51 channels (from the casa split()), so n_chans must be 51
    rt.total_model(a,
                   imres=0.045,
                   nchans=n_chans[DI],
                   chanmin=chanmins[DI],
                   chanstep=chanstep,
                   distance=389,
                   xnpix=256,
                   # vsys=vsys[DI],
                   vsys=v_sys,
                   PA=PA,
                   offs=[pos_x, pos_y],
                   modfile=outputPath,
                   isgas=True,
                   flipme=False,
                   freq0=restfreq,
                   Jnum=jnum,
                   obsv=obsv,
                   hanning=True)

    print("MakeModel() completed")


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
    print("Deleted .im, .uvf, and .vis\n")


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











# The End
