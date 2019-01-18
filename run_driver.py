"""Cail's mcmc.py run wrapper.

"""

# Import some python packages
import os
import pickle
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
from tools import remove, already_exists
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
# Note that these are only the params that are line-independent.
# Line-dependent stuff gets added in at the beginnning of lnprob()
param_dict = {
    'r_out_A':              400,             # AU
    'r_out_B':              200,             # AU
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
    'm_disk_A':            -1.10791,          # Disk Gas Masses (log10 solar masses)
    'm_disk_B':            -1.552842,         # Disk Gas Masses (log10 solar masses)
    'm_stars':             [3.5, 0.4],        # Solar masses (Disk A, B)
    'column_densities':    [1.3e21/(1.59e21), 1e30/(1.59e21)],  # Low, high
    'surf_dens_str_A':     1.,                # Surface density power law index
    'surf_dens_str_B':     1.,                # Surface density power law index
    'v_turb':              0.081,             # Turbulence velocity
    'vert_temp_str':       70.,               # Zq in Kevin's docs
    'r_crit':              100.,              # Critical radius (AU)
    'rot_hands':           [-1, -1],          # disk rotation direction
    'distance':            389.143,           # parsec, errors of 3ish
    'imres':               0.045,             # arcsec/pixel
    'imwidth':             256,               # width of image (pixels)
    }





def main():
    """Establish and evaluate some custom argument options.

    This is called when we run the whole thing: run_driver.py -r does a run.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''Python commands associated with emcee run25, which has 50 walkers and varies the the parameters given param_info.''')

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

        # Save out the param dict for accessing when we want
        pickle.dump(param_dict, open(run_path + 'param_dict.pkl', 'wb'))

        mcmc.run_emcee(run_path=run_path,
                       run_name=today,
                       nsteps=nsteps,
                       nwalkers=nwalkers,
                       lnprob=lnprob #,
                       # param_info=param_info
                       )
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
    #print "Fitting disk 1"
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


# Define likelehood functions
def lnprob(theta, run_name, param_info, mol=mol):
    """Evaluate a set of parameters by making a model and getting its chi2.

    From the emcee docs: a function that takes a vector in the
    parameter space as input and returns the natural logarithm of the
    posterior probability for that position.

    Args:
        theta (list): The proposed steps for each parameter, given by emcee.
        run_name (str): name of run's home directory.
        param_info (list): a single list of tuples of parameter information.
                            Organized as (d1_p0,...,d1_pN, d2_p0,...,d2_pN)
                            and with length = total number of free params
    """
    # Start off by adding in the line-dependent values to param_dict.
    vsys, restfreq, freqs, obsv, chanstep, n_chans, chanmins, jnum = obs_stuff(mol)
    param_dict['offsets']    = [pos_A, pos_B]    # from center (")
    param_dict['offsets']    = [pos_A, pos_B]    # from center (")
    param_dict['vsys']       = vsys              # km/s
    param_dict['restfreq']   = restfreq	   	     # GHz
    param_dict['obsv']       = obsv              # km/s?
    param_dict['jnum']       = lines[mol]['jnum']
    param_dict['chanstep']   = (1) * np.abs(obsv[1] - obsv[0])
    param_dict['chanmins']   = chanmins
    param_dict['nchans']     = n_chans
    param_dict['mol']        = mol


    # Check that the proposed value, theta, is within priors for each var.
    for i, free_param in enumerate(param_info):
        # print '\n', i, free_param
        lower_bound, upper_bound = free_param[-1]
        # If it is, put it into the dict that make_fits calls from
        if lower_bound < theta[i] < upper_bound:
            # print "Taking if"
            name = free_param[0]
            param_dict[name] = theta[i]
            #if name == 'mol_abundance_A' or name == 'mol_abundance_B':
                #print name, theta[i], param_dict[name]
        else:
            # print "Taking else, returning -inf"
            return -np.inf


    # Set up an observation
    obs = fitting.Observation(mol=mol)

    # Make model and the resulting fits image
    model_name = run_name + '_' + str(np.random.randint(1e10))
    model = fitting.Model(observation=obs,
                          run_name=run_name,
                          model_name=model_name)


    # print "Evaluating lnprob for", model_name


    # Make the actual model fits files.
    # make_fits(model, param_info, mol)
    make_fits(model, param_dict, mol)

    model.obs_sample()
    model.chiSq()
    model.delete()
    lnp = -0.5 * sum(model.raw_chis)
    print "Lnprob val: ", lnp
    print '\n'
    return lnp


# This still needs a bunch of cleaning up.
def make_best_fits(run, mol):
    """Do some modeling stuff.

    Args:
        run (mcmc.MCMCrun): the
    """
    # run.main is the pd.df that gets read in from the chain.csv
    subset_df = run.main  # [run.main['r_in'] < 15]

    # Locate the best fit model from max'ed lnprob.
    max_lnp = subset_df['lnprob'].max()
    model_params = subset_df[subset_df['lnprob'] == max_lnp].drop_duplicates()
    print 'Model parameters:\n', model_params.to_string(), '\n\n'

    for param in model_params.columns[:-1]:
        param_dict[param] = model_params[param].values

    disk_params = param_dict.values()

    # intialize model and make fits image
    print 'Making model...'
    # This obviously has to be generalized.
    dataPath = './data/' + mol + '/' + mol + '-short' + lines[mol][min_baseline]
    model = fitting.Model(observation=dataPath,
                          run_name=run_path,
                          model_name=model_name)
    make_fits(model, disk_params)

    print 'Sampling and cleaning...'
    paths = []
    for pointing, rms, starflux in zip(model.observations, aumic_fitting.band6_rms_values[:-1], starfluxes):
        ids = []
        for obs in pointing:
            fix_fits(model, obs, starflux)

            ids.append('_' + obs.name[12:20])
            model.obs_sample(obs, ids[-1])
            model.make_residuals(obs, ids[-1])

        cat_string1 = ','.join([model.path+ident+'.vis' for ident in ids])
        cat_string2 = ','.join([model.path+ident+'.residuals.vis' for ident in ids])
        paths.append('{}_{}'.format(model.path, obs.name[12:15]))

        sp.call(['uvcat', 'vis={}'.format(cat_string2), 'out={}.residuals.vis'.format(paths[-1])], stdout=open(os.devnull, 'wb'))

        sp.call(['uvcat', 'vis={}'.format(cat_string1), 'out={}.vis'.format(paths[-1])], stdout=open(os.devnull, 'wb'))

        model.clean(paths[-1] + '.residuals', rms, show=False)
        model.clean(paths[-1], rms, show=False)

    cat_string1 = ','.join([path + '.vis' for path in paths])
    cat_string2 = ','.join([path + '.residuals.vis' for path in paths])

    sp.call(['uvcat', 'vis={}'.format(cat_string1), 'out={}_all.vis'.format(model.path)], stdout=open(os.devnull, 'wb'))

    sp.call(['uvcat', 'vis={}'.format(cat_string2), 'out={}_all.residuals.vis'.format(model.path)], stdout=open(os.devnull, 'wb'))

    model.clean(model.path + '_all',
                aumic_fitting.band6_rms_values[-1],
                show=False)
    model.clean(model.path + '_all.residuals',
                aumic_fitting.band6_rms_values[-1],
                show=False)

    paths.append('{}_all'.format(model.path))

    print('Making figure...')

    fig = plotting.Figure(layout=(1, 3),
                          paths=[aumic_fitting.band6_fits_images[-1],
                                 paths[-1] + '.fits',
                                 paths[-1] + '.residuals.fits'],
                          rmses=3*[aumic_fitting.band6_rms_values[-1]],
                          texts=[[[4.6, 4.0, 'Data']],
                                 [[4.6, 4.0, 'Model']],
                                 [[4.6, 4.0, 'Residuals']]
                                 ],
                          title=r'Run 6 Global Best Fit Model & Residuals',
                          savefile=run.name + '/' + run.name + '_bestfit_concise.pdf',
                          show=True)


def label_fix(run):
    """Convert parameter labels to latex'ed beauty for plotting.

    This is still Cail's work. I have to go in and make it relevant to my stuff
    """
    """
    for df in [run.main, run.groomed]:

        df.loc[:, 'd_r'] += df.loc[:, 'r_in']
        try:
            df.loc[:, 'starflux'] *= 1e6
        except:
            df.loc[:, 'mar_starflux'] *= 1e6
            df.loc[:, 'jun_starflux'] *= 1e6
            df.loc[:, 'aug_starflux'] *= 1e6

        df.loc[:, 'inc'].where(df.loc[:, 'inc'] < 90,
                               180-df.loc[:, 'inc'], inplace=True)

        df.rename(inplace=True, columns={
            'm_disk': r'$\log \ M_{dust}$ ($M_{\odot}$)',
            'sb_law': r'$p$',
            'scale_factor': r'$h$',
            'r_in': r'$r_{in}$ (au)',
            'd_r': r'$r_{out}$ (au)',
            # 'd_r' : r'$\Delta r$ (au)',
            'inc': r'$i$ ($\degree$)',
            'pa': r'PA  ($\degree$)',
            'mar_starflux': r'March $F_{*}$ ($\mu$Jy)',
            'aug_starflux': r'August $F_{*}$ ($\mu$Jy)',
            'jun_starflux': r'June $F_{*}$ ($\mu$Jy)'})
            """
    return 'Not a functional function.'

#"""
if __name__ == '__main__':
    main()
#"""
