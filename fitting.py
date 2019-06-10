"""Create Observation and Model class that can be read in other files."""

import os
import numpy as np
import subprocess as sp
import matplotlib.pyplot as plt
from astropy.io import fits
from constants import lines
# from tools import plot_fits


class Observation:
    """
    Make the whole observation/data processing shindig a Class.

    This incorporates everything from the path to the original data file to
    the final model. Running it will grab the appropriate data files and
    spit out a cleaned image and some other stuff.
    """

    def __init__(self, mol, cut_baselines=True):
        """Give some init values.

        Args:
            root (str): the name of the directory to source the data files from
            name (str): the name of the data files to grab from root
            rms (float): the rms noise of that particular observation
        """
        if cut_baselines:
            fname_end = '-short' + str(lines[mol]['baseline_cutoff'])
        else:
            fname_end = ''

        self.mol = mol
        self.path = './data/' + mol + '/' + mol + fname_end
        self.uvf  = fits.open(self.path + '.uvf')
        self.fits = fits.open(self.path + '.fits')
        self.baseline_cutoff = 110

        # We manually get rms later on so maybe don't need this?
        self.rms = lines[mol]['rms']
        self.restfreq = lines[mol]['restfreq']

        """
        Not convinced about this stuff. It's not working for my files.
        try:
            self.dec = self.uvf[0].data['OBSDEC'][0]
            self.ra = self.uvf[0].data['OBSRA'][0]
        except:
            self.dec = self.uvf[3].data['DECEPO'][0]
            self.ra = self.uvf[3].data['RAEPO'][0]

        # Keep digging for these guys. They're probably somewhere.
        """

    def clean(self, show=True):
        """
        Clean and image (if desired) some data.

        Note that this is pulled directly from iorek/jonas/.../tools.py/icr()
        """
        # Set observation-specific clean filepath; clear filepaths
        sp.call('rm -rf {}.{{mp,bm,cl,cm}}'.format(self.path), shell=True)

        # Add in the appropriate restfrequency to the header
        sp.call(['puthd',
                 'in={}.vis/restfreq'.format(self.path),
                 'value={}'.format(self.restfreq)
                 ])

        # See June 7 & 22 notes and baseline_cutoff.py for how 30 klambda was
        # determined in select=-uvrange(0,30)
        sp.call(['invert',
                 'vis={}.vis'.format(self.path),
                 'map={}.mp'.format(self.path),
                 'beam={}.bm'.format(self.path),
                 'options=systemp',
                 'select=-uvrange(0,{})'.format(self.baseline_cutoff),
                 'cell=0.045',
                 'imsize=256',
                 'robust=2'
                 ])

        sp.call(['clean',
                 'map={}.mp'.format(self.path),
                 'beam={}.bm'.format(self.path),
                 'out={}.cl'.format(self.path),
                 'niters=10000',
                 'threshold=1e-3'
                 ])

        sp.call(['restor',
                 'map={}.mp'.format(self.path),
                 'beam={}.bm'.format(self.path),
                 'model={}.cl'.format(self.path),
                 'out={}.cm'.format(self.path)
                 ])


        # Display clean image with 2,4,6 sigma contours, if desired
        if show:

            # Display an unimportant imaage to get around the fact that the
            # first image displayed with cgdisp in a session can't be deleted
            sp.call(['cgdisp', 'in=cgdisp_start.im', 'type=p', 'device=/xs'])

            # Get rms for countours
            imstat_out = sp.check_output(['imstat',
                                          'in={}.cm'.format(self.path),
                                          "region='boxes(256,0,512,200)'"])
            clean_rms = float(imstat_out[-38:-29])
            print(("Clean rms is {}".format(clean_rms)))

            # Display
            sp.call(['cgdisp',
                     'in={}.cm,{}.cm'.format(self.path, self.path),
                     'type=p,c', 'device=/xs',
                     'slev=a,{}'.format(clean_rms),
                     'levs1=-6,-4,-2,2,4,6',
                     'region=arcsec,box(-5,-5,5,5)',
                     'labtyp=arcsec',
                     'beamtyp=b,l,3'
                     ])


class Model:
    """Make the model class.

    This takes in an observation class (?) and runs the appropriate modelling
    stuff on it.
    """

    def __init__(self, observation, run_name, model_name, testing=False):
        """Feed the model what it needs.

        Args:
            observation: Observation object holding the data
            run_name: Probably just today. Defined in run_driver.py(33:41)
            model_name: run_name + unique ID number
        """

        if not testing:
            self.modelfiles_path = './mcmc_runs/' + run_name + '/model_files/' + model_name
        else:
            # This is super janky. Basically uses run_name as a path instead.
            # i.e. './test_files/make_fits_test-jan22-1/', 'mf_test'
            self.modelfiles_path = run_name + model_name
        self.observation = observation
        self.raw_chis = []
        self.reduced_chis = []

        # self.delete()  # delete any preexisting files that will conflict

    def delete(self):
        """Delete anything with this path.
        Maybe can do this with pathlib2."""
        sp.call('rm -rf {}*'.format(self.modelfiles_path), shell=True)

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
                 'vis={}.vis'.format(self.observation.path),
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

    def clean(self, path, rms, show=True):
        """Clean (and image) a model.

        Takes in a vis, cleans it, and
        """
        # Set observation-specific clean filepath; clear filepaths
        sp.call('rm -rf {}.{{mp,bm,cl,cm}}'.format(path), shell=True)

        # This is technically redundant with make_fits, but worth doing jic
        sp.call(['puthd',
                 'in={}.vis/restfreq'.format(self.modelfiles_path),
                 'value={}'.format(self.restfreq)
                 ])

        # See June 7 & 22 notes and baseline_cutoff.py for how 30 klambda was
        # determined in select=-uvrange(0,30)
        sp.call(['invert',
                 'vis={}.vis'.format(self.modelfiles_path),
                 'map={}.mp'.format(self.modelfiles_path),
                 'beam={}.bm'.format(self.modelfiles_path),
                 'options=systemp',
                 'select=-uvrange(0,{})'.format(self.baseline_cutoff),
                 'cell=0.045',
                 'imsize=256',
                 'robust=2'
                 ])

        # Maybe use imstat to grab the actual RMS instead of this rough one
        sp.call(['clean',
                 'map={}.mp'.format(self.modelfiles_path),
                 'beam={}.bm'.format(self.modelfiles_path),
                 'out={}.cl'.format(self.modelfiles_path),
                 'niters=10000',
                 'threshold=1e-3'
                 ])

        sp.call(['restor',
                 'map={}.mp'.format(self.modelfiles_path),
                 'beam={}.bm'.format(self.modelfiles_path),
                 'model={}.cl'.format(self.modelfiles_path),
                 'out={}.cm'.format(self.modelfiles_path)
                 ])

        """CAILS CLEAN STUFF:
            # stdout means pipe everything that's not an error into
            # a file. os.devnull is a standard, deleted thing.
            # Clean down to half the observation rms
            sp.call(['invert',
                     'vis={}.vis'.format(path),
                     'map={}.mp'.format(path),
                     'beam={}.bm'.format(path),
                     # 'fwhm=0.91',
                     'cell=0.03arcsec',
                     'imsize=512',
                     'options=systemp,mfs',
                     'robust=2'],
                    stdout=open(os.devnull, 'wb'))
           # imstat_out=sp.check_output(['imstat',
           #     'in={}.mp'.format(path),
           #     "region='boxes(256,0,512,200)'"])
           # dirty_rms = float(imstat_out[-38:-29])
           # print("Dirty rms is {} for {}".format(dirty_rms, path))

            sp.call(['clean',
                     'map={}.mp'.format(path),
                     'beam={}.bm'.format(path),
                     'out={}.cl'.format(path),
                     'niters=10000', 'cutoff={}'.format(rms/2.)],
                    stdout=open(os.devnull, 'wb')
                    )
            sp.call(['restor',
                     'map={}.mp'.format(path),
                     'beam={}.bm'.format(path),
                     'model={}.cl'.format(path),
                     'out={}.cm'.format(path)],
                    stdout=open(os.devnull, 'wb')
                    )
        """
        imstat_out = sp.check_output(['imstat',
                                      'in={}.cm'.format(path),
                                      "region='boxes(256,0,512,200)'"
                                      ])
        rms = float(imstat_out[-38:-29])
        print(("Clean rms is {} for {}".format(rms, path)))

        # Convert MIRIAD .im image file into fits
        sp.call(['fits',
                 'op=xyout',
                 'in={}.cm'.format(path),
                 'out={}.fits'.format(path)],
                stdout=open(os.devnull, 'wb')
                )

        # Display clean image with 2,4,6 sigma contours, if desired
        if show:

            # Display an unimportant image to avoid the fact that the first
            # image displayed with cgdisp in a session can't be deleted
            sp.call(['cgdisp', 'in=cgdisp_start.im', 'type=p', 'device=/xs'])

            # Display
            sp.call(['cgdisp',
                     'in={}.cm,{}.cm'.format(path, path),
                     'type=p,c', 'device=/xs',
                     'slev=a,{}'.format(rms), 'levs1=-6,-4,-2,2,4,6',
                     'region=arcsec,box(-5,-5,5,5)',
                     'labtyp=arcsec', 'beamtyp=b,l,3', ])
            input('\npress enter when ready to go on:')

    def chiSq(self, mol):
        """Calculate the goodness of fit between data and model."""
        # GET VISIBILITIES
        obs = self.observation
        data_vis = obs.uvf[0].data['data'].squeeze()

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

        self.raw_chis.append(raw_chi)
        self.reduced_chis.append(reduced_chi)
        print("Raw Chi2: ", self.raw_chis)
        #print "Reduced Chi2: ", self.reduced_chis
        return self.raw_chis

    def make_residuals(self, obs, suffix='', show=False):
        """Create model residuals, and clean/display if desired.

        Args:
            obs (Observation class): a loaded observation class.
            suffix (str): the unique name given to this model.
            show (bool): Whether or not to clean and cgdisp this model.
                         Note that residual=True isn't currently a valid arg
                         for clean?
        """
        # Note sure why we need the .format() call here if we're
        # wildcard deleting.
        sp.call('rm -rf *.residuals.vis'.format(self.modelfiles_path + suffix),
                shell=True)

        # Subtract model visibilities from data
        # Outfile is residual visibilities
        sp.call(['uvmodel', 'options=subtract',
                 'vis={}.vis'.format(obs.path),
                 'model={}.im'.format(self.modelfiles_path + suffix),
                 'out={}.residuals.vis'.format(self.modelfiles_path + suffix)],
                stdout=open(os.devnull, 'wb')
                )

        if show:
            self.clean(obs, residual=True)

    def view_fits(self):
        """
        Make channel maps of the image. This doesn't really work right now.
        """
        model = fits.getdata(self.modelfiles_path + '.fits')
        # Choose a channel:
        # plot_fits(model)
        return 'THIS DOESNT WORK'
