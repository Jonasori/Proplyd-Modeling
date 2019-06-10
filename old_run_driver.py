"""
Run the whole MCMC shindig! (Old)

This keeps all the values static and bound for a given molecule. Fixing that
in the current run_driver.py with a class.
"""

# Import some python packages
import os
import argparse
import numpy as np
import subprocess as sp
import matplotlib.pyplot as plt
from astropy.constants import M_sun
from astropy.io import fits
# from pathlib2 import Path
# plt.switch_backend('agg')
M_sun = M_sun.value

# Import some files from Kevin's modeling code
from disk_model3 import raytrace as rt
from disk_model3.disk import Disk

# Import some local files
import mcmc
import fitting
# import plotting
from tools import remove, already_exists
from constants import obs_stuff, lines, today, offsets, mol
# Unfortunately, we have to import mol as a global var because manually
# entering it for each run in parallel is shit.


nwalkers = 50
nsteps = 500



# Give the run a name. Exactly equivalent to grid_search.py(250:258)
run_name = today + '-' + mol
run_name_basename = run_name
run_path = './mcmc_runs/' + run_name_basename + '/'
counter = 2
while already_exists(run_path) is True:
    run_name = run_name_basename + '-' + str(counter)
    run_path = './mcmc_runs/' + run_name + '/'
    counter += 1

print(run_path)

run_w_pool = True

pos_A, pos_B = [offsets[0][0], offsets[0][1]], [offsets[1][0], offsets[1][1]]


# An initial list of parameters needed to make a model.
# These get dynamically updated in lnprob (line 348).
# Nothing that is being fit for can be in a tuple
vsys, restfreq, freqs, obsv, chanstep, n_chans, chanmins, jnum = obs_stuff(mol)

param_dict = {
    'r_out_A':              500,             # AU
    'r_out_B':              400,             # AU
    'atms_temp_A':          300,
    'atms_temp_B':          200,
    'mol_abundance_A':      -4,              # Set these to their CO values because they're
    'mol_abundance_B':      -4,              # fixed for CO but get updated for the others.
    'temp_struct_A':        -0.2,            # Tqq in Kevin's code
    'temp_struct_B':        -0.5,            # Tqq in Kevin's code
    'incl_A':               65.,
    'incl_B':               45,
    'pos_angle_A':          69.7,
    'pos_angle_B':          136.,
    'T_mids':              [15, 15],          # Kelvin
    'r_ins':               1,                 # AU
    'r_ins':               [1, 1],            # AU
    'm_disk_A':            -1.10791,          # Disk Gas Masses (log10 solar masses)
    'm_disk_B':            -1.552842,         # Disk Gas Masses (log10 solar masses)
    'm_stars':             [3.5, 0.4],        # Solar masses (Disk A, B)
    'surf_dens_str_A':     1.,                # Surface density power law index
    'surf_dens_str_B':     1.,                # Surface density power law index
    'v_turb':              0.081,             # Turbulence velocity
    'vert_temp_str':       70.,               # Zq in Kevin's docs
    'r_crit':              100.,              # Critical radius (AU)
    'rot_hands':           [-1, -1],          # disk rotation direction
    'distance':            389.143,           # parsec, errors of 3ish
    'imres':               0.045,             # arcsec/pixel
    'imwidth':             256,               # width of image (pixels)
    'mol':                 mol,
    'vsys':                vsys,              # km/s
    'obsv':                obsv,              # km/s
    'nchans':              n_chans,
    'chanmins':            chanmins,
    'restfreq':            restfreq,	   	  # GHz
    'offsets':             [pos_A, pos_B],    # from center (")
    'chanstep':            chanstep,
    'jnum':                lines[mol]['jnum'],
    'column_densities':    lines[mol]['col_dens'],
    'T_freezeout':         lines[mol]['t_fo']
    }

# model = Model(obs, 'jan22-4', 'jan22-4')
# mol = 'hco'
# testing=False
def make_fits(model, param_dict, mol, testing=False):
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

    This is, I think, pretty much redundant with utils.py/make_model(). Would be
    good to consolidate at some point.
    """

    if testing is True:
        dir_list = model.modelfiles_path.split('/')[:-1]
        dir_to_make = '/'.join(dir_list) + '/'
        sp.call(['mkdir {}'.format(dir_to_make)])


    # Make Disk 1
    print("Entering make fits; exporting to" + model.modelfiles_path)
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
                      param_dict['rot_hands'][DI]],
              rtg=False)
    d1.Tco = param_dict['T_freezeout']
    d1.set_rt_grid()
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
                   isgas=True,
                   hanning=True
                   )


    # Now do Disk 2
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
                      param_dict['rot_hands'][DI]],
              rtg=False)
    d2.Tco = param_dict['T_freezeout']
    d2.set_rt_grid()
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
                   isgas=True,
                   hanning=True
                   )

    #print "Both disks have been made; going to summing them now."
    # Now sum those two models, make a header, and crank out some other files.
    a = fits.getdata(model.modelfiles_path + '-d1.fits')
    b = fits.getdata(model.modelfiles_path + '-d2.fits')

    # The actual disk summing
    sum_data = a + b

    # Create the empty structure for the final fits file and insert the data.
    im = fits.PrimaryHDU()
    # The actual disk summing
    im.data = a + b

    # Add the header by modifying a model header.
    with fits.open(model.modelfiles_path + '-d1.fits') as model_fits:
        model_header = model_fits[0].header
    im.header = model_header

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
    im.writeto(model.modelfiles_path + '.fits', overwrite=True)

    remove([model.modelfiles_path + '-d1.fits',
            model.modelfiles_path + '-d2.fits'])



def lnprob(theta, run_name, param_info, mol, multi_fit,
           posang_prior_A=True, posang_prior_B=True):
    """
    Evaluate a set of parameters by making a model and getting its chi2.

    From the emcee docs: a function that takes a vector in the
    parameter space as input and returns the natural logarithm of the
    posterior probability for that position.

    Args:
        theta (list): The proposed steps for each parameter, given by emcee.
        run_name (str): name of run's home directory.
        param_info (list): a single list of tuples of parameter information.
                            Organized as (d1_p0,...,d1_pN, d2_p0,...,d2_pN)
                            and with length = total number of free params
        mol (str):
        posang_prior_X (bool):
        multi_fit (bool): If we want to do a multi-line fit (right now just
                          set up for HCO and HCN since CO is garb), declare it
                          here. Doing so makes the mol argument irrelevant.
    """

    # Let's put priors on both position angles, using Williams et al values.
    # posang_prior_A, posang_prior_B = True, True


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



    # Do the whole observation/model/chi2 process thing.
    # Making this a function to make the single/multi-line fit thing cleaner.
    def get_model_chi(mol):
        obs = fitting.Observation(mol=mol)

        # Make model and the resulting fits image
        model_name = run_name + '_' + str(np.random.randint(1e10))
        model = fitting.Model(observation=obs,
                              run_name=run_name,
                              model_name=model_name)


        # Make the actual model fits files.
        make_fits(model, param_dict, mol)
        model.obs_sample()
        model.chiSq(mol)
        model.delete()
        lnlikelihood = -0.5 * sum(model.raw_chis)
        return lnlikelihood

    # If we want to do a multi-line fit, do it here.
    if multi_fit is true:
        lnlikelihood = -0.5 * sum([get_model_chi(m) for m in ['hco', 'hcn']])
    else:
        lnlikelihood = -0.5 * sum(get_model_chi(mol))


    # Gaussian prior on position angle
    if posang_prior_A:
        mu_posangA = param_dict['pos_angle_A'] - 69.7
        sig_posangA = 1.4 # standard deviation on prior
        # Wikipedia Normal Dist. PDF for where this comes from
        lnprior_posangA = -np.log(np.sqrt(2 * np.pi * sig_posangA**2)) \
                          - mu_posangA**2 / (2 * sig_posangA**2)
    else:
        lnprior_posangA = 0.0

    if posang_prior_B:
        mu_posangB = param_dict['pos_angle_B'] - 135.
        sig_posangB = 15.    # standard deviation on prior
        lnprior_posangB = -np.log(np.sqrt(2 * np.pi * sig_posangB**2)) \
                          - mu_posangB**2 / (2 * sig_posangB**2)
    else:
        lnprior_posangB = 0.0



    # Subtracting (not *ing) because its prior*likelihood -> ln(prior) + ln(likelihood)
    lnprob = lnlikelihood + lnprior_posangA + lnprior_posangB

    print("Lnprob val: ", lnprob)
    print('\n')
    return lnprob


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





## SCHWIMMBAD EXAMPLE
# import math
# def worker(task):
#     a, b = task
#     return math.cos(a) + math.sin(b)
#
# def main(pool):
#     # Here we generate some fake data
#     import random
#     a = [random.uniform(0, 2*math.pi) for _ in range(10000)]
#     b = [random.uniform(0, 2*math.pi) for _ in range(10000)]
#
#     tasks = list(zip(a, b))
#     results = pool.map(worker, tasks)
#     pool.close()
#     # Now we could save or do something with the results object
#
# if __name__ == "__main__":
#     import schwimmbad
#
#     from argparse import ArgumentParser
#     parser = ArgumentParser(description="Schwimmbad example.")
#
#     group = parser.add_mutually_exclusive_group()
#     group.add_argument("--ncores", dest="n_cores", default=1,
#                        type=int, help="Number of processes (uses "
#                                       "multiprocessing).")
#     group.add_argument("--mpi", dest="mpi", default=False,
#                        action="store_true", help="Run with MPI.")
#     args = parser.parse_args()
#
#     pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
#     main(pool)



## JONAS
# def main():
#
#     print("Starting run:" +  run_path + run_name)
#     parser = argparse.ArgumentParser(description="Add cores.")
#     group = parser.add_mutually_exclusive_group()
#
#     parser.add_argument("--ncores", dest="n_cores", default=4,
#                        type=int, help="Number of processes (uses "
#                                       "multiprocessing).")
#     args = parser.parse_args()
#
#     print("This run will have {} walkers taking {} steps each.".format(str(nsteps),
#                                                                        str(nwalkers)))
#     print('and will be distributed over {} cores.\n\n\n'.format(args.n_cores))
#
#     # Set up the parallelization
#     # This is maybe bad. These are two fundamentally different ways of doing
#     # this, but is a temp option.
#     from emcee import __version__ as emcee_version
#     if emcee_version[0] == '2':     # Linux boxes are on v2, cluster is v3
#         from emcee.utils import MPIPool
#         pool = MPIPool()
#
#         # Tell the difference between master and worker processes
#         if not pool.is_master():
#             pool.wait()
#             sys.exit(0)
#     else:
#         from schwimmbad import MultiPool
#         pool = MultiPool(args.n_cores)
#
#
#     mcmc.run_emcee(run_path=run_path,
#                    run_name=run_name,
#                    mol=mol,
#                    nsteps=nsteps,
#                    nwalkers=nwalkers,
#                    lnprob=lnprob,
#                    # param_info=param_info,
#                    pool=pool
#                    )
#     pool.close()
#
#
# if __name__ == "__main__":
#     main()






def main():
    """Establish and evaluate some custom argument options.

    This is called when we run the whole thing: run_driver.py -r does a run.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''Blah''')

    parser.add_argument('-r_ml', '--run_multiline', action='store_true',
                        help='begin or resume eemcee run.')

    parser.add_argument('-r_s', '--run_singleline', action='store_true',
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

    if args.run_multiline or args.run_singleline:
        print("Starting run:" +  run_path + run_name +\
              "\nwith {} steps and {} walkers.".format(str(nsteps), str(nwalkers)))
        print('\n\n\n')


        # Set up the parallelization
        # This is maybe bad. These are two fundamentally different ways of doing
        # this, but is a temp option.
        from emcee import __version__ as emcee_version
        if emcee_version[0] == '2':     # Linux boxes are on v2, cluster is v3
            print("Emcee v2; we're on a Linux box.")
            from emcee.utils import MPIPool
            pool = MPIPool()

            # Tell the difference between master and worker processes
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
        else:
            print("Emcee v3; we're on the cluster.")
            from schwimmbad import MultiPool
            pool = MultiPool(args.n_cores)

        multi_fit = True if args.run_multiline else False
        mcmc.run_emcee(run_path=run_path,
                       run_name=run_name,
                       mol=mol,
                       nsteps=nsteps,
                       nwalkers=nwalkers,
                       lnprob=lnprob,
                       # param_info=param_info
                       pool=pool,
                       multi_fit=multi_fit)
        pool.close()

    else:
        if already_exists(run_path) is False:
            return 'Go in and specify which run you want.'

        run = mcmc.MCMCrun(run_path,
                           run_name,
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




if __name__ == '__main__':
    main()




# The End
