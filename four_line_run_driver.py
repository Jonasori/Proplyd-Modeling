"""Execute an MCMC run


From Meredith:
- Common temperature structure (may have to vary zq?  try fixed first?)
- Vary abundance and radius for each molecule separately
- Keep disk geometry the same for all molecules
- Sum raw chi^2 for each of four molecules

If you don't get a reasonable solution, check what Sam did, but I think maybe we wound up having to vary T_atm separately for each molecule?  Important to try with a consistent temperature structure first so that we can find out how badly it does.

Also check Sam's paper to see if there are descriptions of number density thresholds for upper/lower molecular boundaries, based on photodissociation cross-sections and/or freeze-out.
"""

# Import some python packages
import os
import argparse
import numpy as np
import subprocess as sp
import matplotlib.pyplot as plt
from astropy.constants import M_sun
from astropy.io import fits
plt.switch_backend('agg')
M_sun = M_sun.value

# Import some local files
import mcmc
import fitting
import plotting
from tools import remove, already_exists, already_exists_old
from constants import mol, obs_stuff, lines, today, offsets

# Import some files from Kevin's modeling code
from disk_model import raytrace as rt
from disk_model.disk import Disk

nwalkers = 50
nsteps = 400


# Give the run a name. Exactly equivalent to grid_search.py(250:258)
run_name = today
run_name_basename = run_name
run_path = './mcmc_runs/' + today + '/'
counter = 2
while already_exists(run_path) is True:
    run_name = run_name_basename + '-' + str(counter)
    run_path = './mcmc_runs/' + run_name + '/'

    counter += 1


run_w_pool = True

# I hid the process of getting these bc they're ugly.
vsys, restfreq, freqs, obsv, chanstep, n_chans, chanmins, jnum = obs_stuff(mol)
pos_A, pos_B = [offsets[0][0], offsets[0][1]], [offsets[1][0], offsets[1][1]]

# A list of parameters needed to make a model. These get dynamically updated
# in lnprob (line 348). Nothing that is being fit for can be in a tuple
param_dict = {
    'r_out_A':              400,        # AU
    'r_out_B':              200,        # AU
    'atms_temp_A':          300,
    'atms_temp_B':          200,
    'mol_abundance_A':      -10,
    'mol_abundance_B':      -10,
    'temp_struct_A':        -0.2,
    'temp_struct_B':        -0.2,
    'incl_A':               65.,
    'incl_B':               45,
    'pos_angle_A':          69.7,
    'pos_angle_B':          136.,
    'T_mids':              [15, 15],          # Kelvin
    'r_ins':               1,                 # AU
    'r_ins':               [1, 1],            # AU
    'T_freezeout':         19,                # Freezeout temperature
    'm_disk_A':            -1.10791,        # Disk Gas Masses (log10 solar masses)
    'm_disk_B':            -1.552842,       # Disk Gas Masses (log10 solar masses)
    'm_stars':             [3.5, 0.4],        # Solar masses (Disk A, B)
    'column_densities':    [1.3e21/(1.59e21), 1e30/(1.59e21)],  # Low, high
    'surf_dens_str_A':     1.,                # Surface density power law index
    'surf_dens_str_B':     1.,                # Surface density power law index
    'v_turb':              0.081,             # Turbulence velocity
    'vert_temp_str':       70.,               # Zq in Kevin's docs
    'r_crit':              100.,              # Critical radius (AU)
    'rot_hands':           [-1, -1],          # disk rotation direction
    'distance':            389.143,           # parsec, errors of 3ish
    'offsets':             [pos_A, pos_B],    # from center (")
    'offsets':             [pos_A, pos_B],    # from center (")
    'vsys':                vsys,              # km/s
    'restfreq':            restfreq,		  # GHz
    'obsv':                obsv,              # km/s?
    'jnum':                lines[mol]['jnum'],
    'chanstep':            (1) * np.abs(obsv[1] - obsv[0]),
    'chanmins':            chanmins,
    'nchans':              n_chans,
    'imres':               0.045,             # arcsec/pixel
    'imwidth':             256                # width of image (pixels)
    }


# Note that this is what is fed to MCMC to dictate how the walkers move, not
# the actual set of vars that make_fits pulls from.
# The order matters here (for comparing in lnprob)
# Note that param_info is of form:
# [param name, init_pos_center, init_pos_sigma, (prior lower, prior upper)]
if mol != 'co':
    param_info = [('atms_temp_A',       300,     150,      (0, np.inf)),
                  ('temp_struct_A',    -0.,      1.,       (-3., 3.)),
                  ('incl_A',            65.,     30.,      (0, 90.)),
                  ('pos_angle_A',       70,      45,       (0, 360)),
                  ('atms_temp_B',       200,     150,      (0, np.inf)),
                  ('mol_abundance_B',   -8,      3,        (-13, -3)),
                  ('temp_struct_B',     0.,      1,        (-3., 3.)),
                  ('incl_B',            45.,     30,       (0, 90.)),
                  ('pos_angle_B',       136.0,   45,       (0, 360)),

                  ('r_out_A-cs',        500,     300,      (10, 1000)),
                  ('r_out_A-co',        500,     300,      (10, 1000)),
                  ('r_out_A-hco',       500,     300,      (10, 1000)),
                  ('r_out_A-hcn',       500,     300,      (10, 1000)),

                  ('r_out_B-cs',        500,     300,      (10, 1000)),
                  ('r_out_B-co',        500,     300,      (10, 1000)),
                  ('r_out_B-hco',       500,     300,      (10, 1000)),
                  ('r_out_B-hcn',       500,     300,      (10, 1000))

                  ('mol_abundance_A-cs',  -8,      3,        (-13, -3)),
                  ('mol_abundance_A-co',  -8,      3,        (-13, -3)),
                  ('mol_abundance_A-hco', -8,      3,        (-13, -3)),
                  ('mol_abundance_A-hcn', -8,      3,        (-13, -3)),

                  ('mol_abundance_B-cs',  -8,      3,        (-13, -3)),
                  ('mol_abundance_B-co',  -8,      3,        (-13, -3)),
                  ('mol_abundance_B-hco', -8,      3,        (-13, -3)),
                  ('mol_abundance_B-hcn', -8,      3,        (-13, -3)),

                  ]

else:
    param_info = [('r_out_A',           500,     300,      (10, 1000)),
                  ('atms_temp_A',       300,     150,      (0, np.inf)),
                  ('m_disk_A',          -1.,      1.,      (-2.5, 0)),
                  ('temp_struct_A',    -0.,      1.,       (-3., 3.)),
                  ('incl_A',            65.,     30.,      (0, 90.)),
                  ('pos_angle_A',       70,      45,       (0, 360)),
                  ('r_out_B',           500,     300,      (10, 1000)),
                  ('atms_temp_B',       200,     150,      (0, np.inf)),
                  ('m_disk_B',          -1.,      1.,      (-2.5, 0))
                  ('temp_struct_B',     0.,      1,        (-3., 3.)),
                  ('incl_B',            45.,     30,       (0, 90.)),
                  ('pos_angle_B',       136.0,   45,       (0, 360))
                  ]




def main():
    """Establish and evaluate some custom argument options.

    This is called when we run the whole thing: run_driver.py -r does a run.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''Python commands associated with emcee run25, which has 50 walkers and varies the following parameters for each disk:
    1)  outer disk radius
    2)  atmospheric temperature
    3)  relative molecular abundance
    4)  temperature structure power law index (tqq in Kevin's code)
    5)  outer radius (really inner radius + dr)
    6)  inclination
    7)  position angle
    This run is the first of Jonas' runs.''')

    parser.add_argument('-r', '--run', action='store_true',
                        help='begin or resume eemcee run.')

    parser.add_argument('-a', '--analyze', action='store_true',
                        help='analyze sampler chain, producing an evolution plot, corner plot, and image domain figure.')

    parser.add_argument('-b', '--burn_in', default=0, type=int,
                        help='number of steps \'burn in\' steps to exclude')

    parser.add_argument('-bf', '--best_fit', action='store_true',
                        help='generate best fit model images and residuals')

    parser.add_argument('-con', '--concise', action='store_true',
                        help='concise best fit')

    parser.add_argument('-c', '--corner', action='store_true',
                        help='generate corner plot')

    parser.add_argument('-cvars', '--corner_vars', default=None, nargs='+',
                        help='concise best fit')

    parser.add_argument('-e', '--evolution', action='store_true',
                        help='generate walker evolution plot.')

    parser.add_argument('-kde', '--kernel_density', action='store_true',
                        help='generate kernel density estimate (kde) of posterior distribution')


    args = parser.parse_args()

    if args.run:
        print "Starting run:", run_path + today
        print "with " + str(nsteps) + " steps and " + str(nwalkers) + "walkers."
        print '\n\n\n'
        mcmc.run_emcee(run_path=run_path,
                       run_name=today,
                       nsteps=nsteps,
                       nwalkers=nwalkers,
                       lnprob=lnprob,
                       param_info=param_info)
    else:
        if already_exists(run_path) is False:
            return 'Go in and specify which run you want.'

        run = mcmc.MCMCrun(run_path,
                           today,
                           nwalkers=nwalkers,
                           burn_in=args.burn_in)
        # old_nsamples = run.groomed.shape[0]
        # run.groomed = run.groomed[run.groomed['r_in'] + run.groomed['d_r'] > 20]
        # print('{} samples removed.'.format(old_nsamples - run.groomed.shape[0]))

        # This was down in the arg analysis but seems separate.
        # Also, aumic_fitting doesn't seem to exist anymore?
        label_fix(run)

        # Read the arguments passed and execute them.
        if args.corner_vars:
            cols = list(run.groomed.columns)
            col_indices = [cols.index(col) for col in args.corner_vars]

        if args.analyze or args.best_fit:
            make_best_fits(run, concise=args.concise)

        if args.corner_vars:
            args.corner_vars = run.groomed.columns[col_indices]

        if args.analyze or args.evolution:
            run.evolution()

        if args.analyze or args.kernel_density:
            run.kde()

        if args.analyze or args.corner:
            run.corner(variables=args.corner_vars)


def make_fits(model, param_dict, mol=mol, testing=False):
    """Take in two list of disk params and return a two-disk model.

    Args:
        model (Model instance): the model that this should go into.
        param_dict (dict): the dynamically updated, global parameter dictionary
                            of parameters. Each parameter that is different for
                            the two disks is a tuple of [diskA_val, diskB_val]
        testing: if you're setting up this model as a diagnostic, you'll run into
                 problems since the necessary directories are set up in run_emcee.

    Output:
        (model.path).fits
    """

    if testing is True:
        dir_list = model.modelfiles_path.split('/')[:-1]
        dir_to_make = '/'.join(dir_list) + '/'
        sp.call(['mkdir {}'.format(dir_to_make)])

    # Make Disk 1
    print "Entering make fits; exporting to", model.modelfiles_path
    DI = 0
    d1 = Disk(params=[param_dict['temp_struct_A'],
                      10**param_dict['m_disk_A'],
                      param_dict['surf_dens_str_A'],
                      param_dict['r_ins'][DI],
                      param_dict['r_out_A'],
                      param_dict['r_crit'],
                      param_dict['incl_A'],
                      param_dict['m_stars'][DI],
                      10**param_dict['mol_abundance_A'],
                      param_dict['v_turb'],
                      param_dict['vert_temp_str'],
                      param_dict['T_mids'][DI],
                      param_dict['atms_temp_A'],
                      param_dict['column_densities'],
                      [param_dict['r_ins'][DI], param_dict['r_out_A']],
                      param_dict['rot_hands'][DI]
                      ])
    rt.total_model(d1,
                   imres=param_dict['imres'],
                   distance=param_dict['distance'],
                   chanmin=param_dict['chanmins'][DI],
                   nchans=param_dict['nchans'][DI],
                   chanstep=param_dict['chanstep'],
                   flipme=False,
                   Jnum=param_dict['jnum'],
                   freq0=param_dict['restfreq'],
                   xnpix=param_dict['imwidth'],
                   vsys=param_dict['vsys'][DI],
                   PA=param_dict['pos_angle_A'],
                   offs=param_dict['offsets'][DI],
                   modfile=model.modelfiles_path + '-d1',
                   obsv=param_dict['obsv'],
                   isgas=True
                   )

    # Now do Disk 2
    #print "Now fitting disk 2"
    DI = 1
    d2 = Disk(params=[param_dict['temp_struct_B'],
                      10**param_dict['m_disk_B'],
                      param_dict['surf_dens_str_B'],
                      param_dict['r_ins'][DI],
                      param_dict['r_out_B'],
                      param_dict['r_crit'],
                      param_dict['incl_B'],
                      param_dict['m_stars'][DI],
                      10**param_dict['mol_abundance_B'],
                      param_dict['v_turb'],
                      param_dict['vert_temp_str'],
                      param_dict['T_mids'][DI],
                      param_dict['atms_temp_B'],
                      param_dict['column_densities'],
                      [param_dict['r_ins'][DI], param_dict['r_out_B']],
                      param_dict['rot_hands'][DI]
                      ])
    rt.total_model(d2,
                   imres=param_dict['imres'],
                   distance=param_dict['distance'],
                   chanmin=param_dict['chanmins'][DI],
                   nchans=param_dict['nchans'][DI],
                   chanstep=param_dict['chanstep'],
                   flipme=False,
                   Jnum=param_dict['jnum'],
                   freq0=param_dict['restfreq'],
                   xnpix=param_dict['imwidth'],
                   vsys=param_dict['vsys'][DI],
                   PA=param_dict['pos_angle_B'],
                   offs=param_dict['offsets'][DI],
                   modfile=model.modelfiles_path + '-d2',
                   obsv=param_dict['obsv'],
                   isgas=True
                   )

    #print "Both disks have been made; going to summing them now."
    # Now sum those two models, make a header, and crank out some other files.
    a = fits.getdata(model.modelfiles_path + '-d1.fits')
    b = fits.getdata(model.modelfiles_path + '-d2.fits')
    header_info_from_model = fits.open(model.modelfiles_path + '-d1.fits')

    # The actual disk summing
    sum_data = a + b

    # Create the empty structure for the final fits file and insert the data.
    im = fits.PrimaryHDU()
    im.data = sum_data

    # Add the header by modifying a model header.

    # im_header = header_info_from_model[0].header
    # I think this should be an attribute, not a variable.
    im.header = header_info_from_model[0].header
    header_info_from_model.close()

    # Swap out some of the vals using values from the data file used by model:
    # header_info_from_data = fits.open('../data/{}/{}.fits'.format(mol, mol))
    header_info_from_data = model.observation.fits
    data_header = header_info_from_data[0].header
    header_info_from_data.close()

    # Put in RA, Dec and restfreq
    im.header['CRVAL1'] = data_header['CRVAL1']
    im.header['CRVAL2'] = data_header['CRVAL2']
    im.header['RESTFRQ'] = data_header['RESTFREQ']
    # im.header['EPOCH'] = data_header['EPOCH']

    # Write it out to a file, overwriting the existing one if need be
    fitsout = model.modelfiles_path + '.fits'
    #print "Writing out model fits file to: ", fitsout
    im.writeto(fitsout, overwrite=True)

    """
    Not sure why this is commented out; seems like it's important.
    But it's showing up in the header somehow, so I guess no need to put it in.
    sp.call
    puthd in={fitsout.fits}/EPOCH value=2000.0
    """

    # Clear out the individual disk models now that we've got the data.
    remove([model.modelfiles_path + '-d1.fits',
            model.modelfiles_path + '-d2.fits'])



#"""
if __name__ == '__main__':
    main()
#"""