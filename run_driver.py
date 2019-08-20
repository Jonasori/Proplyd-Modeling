"""Run the whole MCMC shindig."""

# Import some python packages
import sys
import yaml
import emcee
import argparse
import numpy as np
import subprocess as sp
from yaml import CLoader, CDumper
# from pathlib2 import Path
# plt.switch_backend('agg')


# Import some local files
import utils
from constants import today
from tools import remove, already_exists


def lnprob(theta, run_path, param_info, mol):
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
        PA_prior_X (bool):

    Big, anticipated problems:
        - run_name/run_path stuff, including in fitting.Model().
        -
    """

    # print('\nTheta:\n{}\n'.format(theta))
    # Check that the proposed value, theta, is within priors for each var.
    # This should work with multi-line.
    for i, free_param in enumerate(param_info):
        lower_bound, upper_bound = free_param[-1]
        # If it is, put it into the dict that make_fits calls from
        if not lower_bound < theta[i] < upper_bound:
            return -np.inf

    # print("Theta: {}".format(theta))


    # Simplify the chi-getting process
    def get_model_chi(mol, param_path, run_path, model_name):
        """Consolidate the actual chi-getting process."""
        # print('Param path: {}\nModel Name: {}'.format(param_path, model_name))
        model = utils.Model(mol, param_path, run_path, model_name)
        model.make_fits()
        model.obs_sample()
        model.chiSq(mol)
        model.delete()
        return model.raw_chi
    # Update the param files appropriately and get the chi-squared values.
    if mol == 'multi':
        param_dicts = {'hco': yaml.load(open(
                            '{}params-hco.yaml'.format(run_path), 'r'),
                                        Loader=CLoader),
                       'hcn': yaml.load(open(
                            '{}params-hcn.yaml'.format(run_path), 'r'),
                                         Loader=CLoader)
                       }
        # Check if it's a mol-specific param, and add in appropriately.
        # There's probably a more elegant way to do this
        for i, free_param in enumerate(param_info):
            name = free_param[0]
            # print(name, theta[i])
            if 'hco' in name:
                param_dicts['hco'][name] = theta[i]
            elif 'hcn' in name:
                param_dicts['hcn'][name] = theta[i]
            else:
                param_dicts['hco'][name] = theta[i]
                param_dicts['hcn'][name] = theta[i]

        # Avoid crashing between param files in parallel w/ unique identifier.
        # Also used as a unique id for the resulting model files.
        unique_id = str(np.random.randint(1e10))
        model_name = 'model_' + unique_id
        param_path_hco = '{}model_files/params-hco_{}.yaml'.format(run_path, unique_id)
        param_path_hcn = '{}model_files/params-hcn_{}.yaml'.format(run_path, unique_id)
        yaml.dump(param_dicts['hco'], open(param_path_hco, 'w+'), Dumper=CDumper)
        yaml.dump(param_dicts['hcn'], open(param_path_hcn, 'w+'), Dumper=CDumper)

        # Get the actual values
        lnlikelihood = -0.5 * sum([get_model_chi('hco', param_path_hco, run_path, model_name),
                                   get_model_chi('hcn', param_path_hcn, run_path, model_name)])

        # This is really inefficient (open the file, write out the modifications,
        # close it, open it for the modeling, then delete it), but I like
        # having get_model_chi() just pull in a param file instead of a dict.
        # Could be changed.
        remove([param_path_hco, param_path_hcn])

        # This is pretty janky, but if at least one of the lines wants
        # Gaussian priors on PA, then just do it.
        PA_prior_A = True if True in (param_dicts['hco']['PA_prior_A'], param_dicts['hcn']['PA_prior_A']) else False
        PA_prior_B = True if True in (param_dicts['hco']['PA_prior_B'], param_dicts['hcn']['PA_prior_B']) else False


    else:  # Single line
        param_dict = yaml.load(open('{}params-{}.yaml'.format(run_path, mol),
                                    'r'), Loader=CLoader)
        for i, free_param in enumerate(param_info):
            name = free_param[0]
            param_dict[name] = theta[i]

        unique_id = str(np.random.randint(1e10))
        model_name = 'model_' + unique_id
        param_path = '{}/model_files/params-{}_{}.yaml'.format(run_path, mol, unique_id)
        yaml.dump(param_dict, open(param_path, 'w+'), Dumper=CDumper)
        lnlikelihood = -0.5 * get_model_chi(mol, param_path, run_path, model_name)
        remove(param_path)


        PA_prior_A, PA_prior_B = param_dict['PA_prior_A'], param_dict['PA_prior_B']


    # Since PA is not fit individually, just grab one of them for ML fits.
    p_dict = param_dicts['hco'] if mol == 'multi' else param_dict
    if PA_prior_A:
        mu_posangA = p_dict['pos_angle_A'] - 69.7
        sig_posangA = 1.4  # standard deviation on prior
        # Wikipedia Normal Dist. PDF for where this comes from
        lnprior_posangA = -np.log(np.sqrt(2 * np.pi * sig_posangA**2)) \
                          - mu_posangA**2 / (2 * sig_posangA**2)
    else:
        lnprior_posangA = 0.0

    if PA_prior_B:
        mu_posangB = p_dict['pos_angle_B'] - 135.
        sig_posangB = 15.    # standard deviation on prior
        lnprior_posangB = -np.log(np.sqrt(2 * np.pi * sig_posangB**2)) \
                          - mu_posangB**2 / (2 * sig_posangB**2)
    else:
        lnprior_posangB = 0.0



    # Subtracting (not *ing) because
    # ln(prior*likelihood) -> ln(prior) + ln(likelihood)
    lnprob = lnlikelihood + lnprior_posangA + lnprior_posangB

    # print("Lnprob val: ", lnprob)
    # print('\n')
    return lnprob



# Could be cool to rewrite this as a class with _call_ taking the place of -r
def run_emcee(mol, lnprob, pool):
    """
    Make an actual MCMC run.

    Other than in setting up param_info, this is actually line-agnostic.
    The line-specificity is created in the lnprob function.

    Args:
        run_path (str): the name to output I guess
        run_name (str): the name to feed the actual emcee routine (line 360)
        mol (str): which line we're running.
        nsteps (int): How many steps we're taking.
        nwalkers (int): How many walkers we're using
        lnprob (func): The lnprob function to feed emcee
        param_info (list): list of [param name,
                                    initial_position_center,
                                    initial_position_sigma,
                                    (prior low bound, prior high bound)]
                            for each parameter.
                            The second two values set the position & size
                            for a random Gaussian ball of initial positions
    """


    # Set up a run naming convension:
    run_name = today + '-' + mol
    run_name_basename = run_name
    run_path = './mcmc_runs/' + run_name_basename + '/'
    counter = 2
    while already_exists(run_path) is True:
        run_name = run_name_basename + '-' + str(counter)
        run_path = './mcmc_runs/' + run_name + '/'
        counter += 1

    print('Run path is {}'.format(run_path))


    print("Setting up directories for new run")
    remove(run_path)
    sp.call(['mkdir', run_path])
    sp.call(['mkdir', run_path + '/model_files'])

    # Make a copy of the initial parameter dict so we can modify it
    if mol is 'multi':
        sp.call(['cp', 'hco.yaml', '{}params-hco.yaml'.format(run_path)])
        sp.call(['cp', 'hcn.yaml', '{}params-hcn.yaml'.format(run_path)])
    else:
        sp.call(['cp', mol + '.yaml', '{}params-{}.yaml'.format(run_path,
                                                                mol)])


 
    # Note that this is what is fed to MCMC to dictate how the walkers move, not
    # the actual set of vars that make_fits pulls from.
    # ORDER MATTERS here (for comparing in lnprob)
    # Values that are commented out default to the starting positions in run_driver/param_dict
    # Note that param_info is of form:
    # [param name, init_pos_center, init_pos_sigma, (prior lower, prior upper)]

    if mol is 'multi':
        # There are more params to fit here.
        param_info = [('r_out_A_hco',           500,     300,      (10, 700)),
                      ('r_out_A_hcn',           500,     300,      (10, 700)),
                      ('atms_temp_A',           200,     150,      (0, 1000)),
                      ('mol_abundance_A_hco',   -8,      3,        (-13, -3)),
                      ('mol_abundance_A_hcn',   -8,      3,        (-13, -3)),
                      # ('mol_abundance_A_cs',    -8,      3,        (-13, -3)),
                      ('temp_struct_A',         -0.,      1.,      (-3., 3.)),
                      # ('incl_A',            65.,     30.,      (0, 90.)),
                      ('pos_angle_A',           70,      45,       (0, 360)),
                      ('r_out_B_hco',           500,     300,      (10, 400)),
                      ('r_out_B_hcn',           500,     300,      (10, 400)),
                      ('atms_temp_B',           200,     150,      (0, 1000)),
                      ('mol_abundance_B_hco',   -8,      3,        (-13, -3)),
                      ('mol_abundance_B_hcn',   -8,      3,        (-13, -3)),
                      # ('mol_abundance_B_cs',    -8,      3,        (-13, -3)),
                      # ('temp_struct_B',     0.,      1,        (-3., 3.)),
                      # ('incl_B',            45.,     30,       (0, 90.)),
                      ('pos_angle_B',           136.0,   45,       (0, 180))
                      ]


    # HCO+, HCN, or CS
    elif mol != 'co':
        param_info = [('r_out_A',           500,     300,      (10, 700)),
                      ('atms_temp_A',       300,     150,      (0, 1000)),
                      ('mol_abundance_A',   -8,      3,        (-13, -3)),
                      ('temp_struct_A',     -0.,      1.,      (-3., 3.)),
                      # ('incl_A',            65.,     30.,      (0, 90.)),
                      ('pos_angle_A',       70,      45,       (0, 360)),
                      ('r_out_B',           500,     300,      (10, 400)),
                      ('atms_temp_B',       200,     150,      (0, 1000)),
                      ('mol_abundance_B',   -8,      3,        (-13, -3)),
                      # ('temp_struct_B',     0.,      1,        (-3., 3.)),
                      # ('incl_B',            45.,     30,       (0, 90.)),
                      ('pos_angle_B',       136.0,   45,       (0, 180))
                      ]
    else:
        param_info = [('r_out_A',           500,     300,      (10, 700)),
                      ('atms_temp_A',       300,     150,      (0, 1000)),
                      ('m_disk_A',          -1.,      1.,      (-4.5, 0)),
                      ('temp_struct_A',     -0.,      1.,      (-3., 3.)),
                      # ('incl_A',            65.,     30.,      (0, 90.)),
                      ('pos_angle_A',        70,      45,      (0, 180)),
                      ('r_out_B',           500,     300,      (10, 400)),
                      ('atms_temp_B',       200,     150,      (0, 1000)),
                      ('m_disk_B',          -4.,      1.,      (-6., 0)),
                      # ('temp_struct_B',     0.,      1,        (-3., 3.)),
                      # ('incl_B',            45.,     30,       (0, 90.)),
                      # ('pos_angle_B',       136.0,   45,       (0, 180))
                      ]


    # Start a new file for the chain; set up a header line
    chain_filename = run_path + run_name + '_chain.csv'
    with open(chain_filename, 'w') as f:
        param_names = [param[0] for param in param_info]
        np.savetxt(f, (np.append(param_names, 'lnprob'), ),
                   delimiter=',', fmt='%s')



    m = 'hco' if mol is 'multi' else mol
    f = yaml.load(open('{}params-{}.yaml'.format(run_path, m), 'r'), Loader=CLoader)
    nwalkers, nsteps = f['nwalkers'], f['nsteps']

    # Set up initial positions
    # randn makes an n-dimensional array of rands in [0,1]
    pos = []
    for i in range(nwalkers):
        pos_walker = []
        for param in param_info:
            pos_i = param[1] + param[2]*np.random.randn()
            # Make sure we're starting within priors
            lower_bound, upper_bound = param[-1]
            while not lower_bound < pos_i < upper_bound:
                pos_i = param[1] + param[2]*np.random.randn()

            pos_walker.append(pos_i)
        pos.append(pos_walker)

    # Initialize sampler chain
    # Recall that param_info is a list of length len(d1_params)+len(d2_params)
    print("Initializing sampler.")
    ndim = len(param_info)


    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(run_path, param_info, mol),
                                    pool=pool)

    # Initialize a generator to provide the data. They changed the arg
    # storechain -> store sometime between v2.2.1 (iorek) and v3.0rc2 (cluster)
    from emcee import __version__ as emcee_version
    # print("About to run sampler")
    if emcee_version[0] == '2':     # Linux boxes are on v2, cluster is v3
        run = sampler.sample(pos, iterations=nsteps, storechain=False)  # for iorek
    else:
        run = sampler.sample(pos, iterations=nsteps, store=False)   # for cluster
    """Note that sampler.sample returns:
            pos: list of the walkers' current positions in an object of shape
                    [nwalkers, ndim]
            lnprob: The list of log posterior probabilities for the walkers at
                    positions given by pos.
                    The shape of this object is (nwalkers, dim)
            rstate: The current state of the random number generator.
            blobs (optional): The metadata "blobs" associated with the current
                              position. The value is only returned if
                              lnpostfn returns blobs too.
            """
    lnprobs = []
    # import pdb; pdb.set_trace()


    # print("THIS CURRENT WORKING DIRECTORY IS" + os.getcwd() + '\n\n')
    # print("About to loop over run")
    for i, result in enumerate(run):
        print("Got a result")
        pos, lnprobs, blob = result

        # Log out the new positions
        with open(chain_filename, 'a') as f:
            new_step = [np.append(pos[k], lnprobs[k]) for k in range(nwalkers)]
            print("Adding a new step to the chain: ", new_step)
            np.savetxt(f, new_step, delimiter=',')

    print("Ended run")




def main():
    """Establish and evaluate some custom argument options.

    This is called when we run the whole thing: run_driver.py -r does a run.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''Blah''')

    parser.add_argument('-hco', '--run_hco', action='store_true',
                        help='Begin an HCO+ run.')
    parser.add_argument('-hcn', '--run_hcn', action='store_true',
                        help='Begin an HCN run.')
    parser.add_argument('-co', '--run_co', action='store_true',
                        help='Begin an CO run.')
    parser.add_argument('-cs', '--run_cs', action='store_true',
                        help='Begin an CS run.')
    parser.add_argument('-multi', '--run_multi', action='store_true',
                        help='Begin a multi-line (HCO+/HCN) run.')

    args = parser.parse_args()



    if args.run_hco or args.run_hcn or args.run_co or args.run_cs or args.run_multi:
        print("Running now")
        # Set up the parallelization
        # This is maybe bad. These are two fundamentally different ways of doing
        # this, but is a temp solution.
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


        # Do the thing
        if args.run_hco:
            run_emcee(mol='hco', lnprob=lnprob, pool=pool)

        if args.run_hcn:
            run_emcee(mol='hcn', lnprob=lnprob, pool=pool)

        if args.run_co:
            run_emcee(mol='co', lnprob=lnprob, pool=pool)

        if args.run_cs:
            run_emcee(mol='cs', lnprob=lnprob, pool=pool)

        if args.run_multi:
            run_emcee(mol='multi', lnprob=lnprob, pool=pool)


        pool.close()





if __name__ == '__main__':
    main()







# The End
# Everything below is leftovers


















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
