"""Run the whole MCMC shindig."""

# Import some python packages
import sys
import yaml, json
import emcee
import argparse
import numpy as np
import pandas as pd
import subprocess as sp
from yaml import CLoader, CDumper
# from pathlib2 import Path
# plt.switch_backend('agg')
import warnings
warnings.filterwarnings("ignore")

# Import some local files
import utils
from constants import today
from tools import remove, already_exists

# See here for info: https://emcee.readthedocs.io/en/stable/tutorials/parallel/#parallel
os.environ["OMP_NUM_THREADS"] = "1"


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
        with open('{}params-hco.json'.format(run_path), 'r') as f_hco:
            with open('{}params-hcn.json'.format(run_path), 'r') as f_hcn:
#                 param_dicts = {'hco': yaml.load(f_hco, Loader=CLoader),
#                                'hcn': yaml.load(f_hcn, Loader=CLoader)
#                                }
                param_dicts = {'hco': json.load(f_hco),
                               'hcn': json.load(f_hcn)
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
     
    
        # This is really inefficient (open the file, write out the modifications,
        # close it, open it for the modeling, then delete it), but I like
        # having get_model_chi() just pull in a param file instead of a dict.
        # Could be changed.
        param_path_hco = '{}model_files/params-hco_{}.json'.format(run_path, unique_id)
        param_path_hcn = '{}model_files/params-hcn_{}.json'.format(run_path, unique_id)
        
        with open(param_path_hco, 'w+') as f_hco:
#             yaml.dump(param_dicts['hco'], f_hco, Dumper=CDumper)
            json.dump(param_dicts['hco'], f_hco)
        with open(param_path_hcn, 'w+') as f_hcn:
#             yaml.dump(param_dicts['hcn'], f_hcn, Dumper=CDumper)
            json.dump(param_dicts['hcn'], f_hcn)

        # Get the actual values
        lnlikelihood = -0.5 * sum([get_model_chi('hco', param_path_hco, run_path, model_name),
                                   get_model_chi('hcn', param_path_hcn, run_path, model_name)])
        remove([param_path_hco, param_path_hcn])

        # This is pretty janky, but if at least one of the lines wants
        # Gaussian priors on PA, then just do it.
        PA_prior_A = True if True in (param_dicts['hco']['PA_prior_A'], param_dicts['hcn']['PA_prior_A']) else False
        PA_prior_B = True if True in (param_dicts['hco']['PA_prior_B'], param_dicts['hcn']['PA_prior_B']) else False


    else:  # Single line
        with open('{}params-{}.json'.format(run_path, mol)) as f:
#             param_dict = yaml.load(f, Loader=CLoader)
            param_dict = json.load(f)
        for i, free_param in enumerate(param_info):
            name = free_param[0]
            param_dict[name] = theta[i]

        unique_id = str(np.random.randint(1e10))
        model_name = 'model_' + unique_id
        param_path = '{}/model_files/params-{}_{}.json'.format(run_path, mol, unique_id)
        with open(param_path, 'w+') as f:
#             yaml.dump(param_dict, f, Dumper=CDumper)
            json.dump(param_dict, f)
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
def run_emcee(mol, lnprob, pool, resume_run=None):
    """
    Make an actual MCMC run.

    Other than in setting up param_info, this is actually line-agnostic.
    The line-specificity is created in the lnprob function.

    Args:
        mol (str): which line we're running.
        lnprob (func): The lnprob function to feed emcee
        pool ():
        from_checkpoint (path): If we want to restart a dead run, give that run's name here
            (i.e. 'nov1-multi'). Assumes runs are located in /Volumes/disks/jonas/modeling/mcmc_runs/
        
        
        
        
        param_info (list): list of [param name,
                                    initial_position_center,
                                    initial_position_sigma,
                                    (prior low bound, prior high bound)]
                            for each parameter.
                            The second two values set the position & size
                            for a random Gaussian ball of initial positions
    """


    if resume_run:
        run_name = resume_run
        run_path = './mcmc_runs/{}/'.format(run_name)
        print("Resuming old run at " + run_path)
    else:
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
            sp.call(['cp', 'params-hco.json', '{}params-hco.json'.format(run_path)])
            sp.call(['cp', 'params-hcn.json', '{}params-hcn.json'.format(run_path)])
        else:
            sp.call(['cp', "params-" + mol + '.json', '{}params-{}.json'.format(run_path,
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

    m = 'hco' if mol is 'multi' else mol
    with open('{}params-{}.json'.format(run_path, m), 'r') as f_base:
#         f = yaml.load(f_base, Loader=CLoader)
        f = json.load(f_base)
        nwalkers, nsteps = f['nwalkers'], f['nsteps']

    # Set up initial positions
    
    if resume_run:
        chain_filename = '/Volumes/disks/jonas/modeling/mcmc_runs/{}/{}_chain.csv'.format(resume_run, resume_run)
        last_step = pd.read_csv(chain_filename).iloc[-nwalkers:]
        # .tolist() makes this into a list in the correct order   
        # This might be backwards? Maybe need .iloc[-i]
        pos = [last_step.iloc[i].tolist() for i in range(nwalkers)]
    
    else:
        # Start a new file for the chain; set up a header line
        chain_filename = run_path + run_name + '_chain.csv'
        with open(chain_filename, 'w') as f:
            param_names = [param[0] for param in param_info]
            np.savetxt(f, (np.append(param_names, 'lnprob'), ),
                       delimiter=',', fmt='%s')

        # randn randomly samples a normal distribution
        pos = []
        for i in range(nwalkers):
            pos_walker = []
            for param in param_info:
                pos_i = float(param[1] + param[2]*np.random.randn())
                # Make sure we're starting within priors
                lower_bound, upper_bound = param[-1]
                while not lower_bound < pos_i < upper_bound:
                    pos_i = float(param[1] + param[2]*np.random.randn())
                pos_walker.append(pos_i)
            pos.append(pos_walker)
#         print("Positions: {}\n\n".format(pos))

    # Initialize sampler chain
    # Recall that param_info is a list of length len(d1_params)+len(d2_params)
    print("Initializing sampler.")
    ndim = len(param_info)
    
    # Emcee v3 seems cool. Should upgrade: https://emcee.readthedocs.io/en/stable/user/upgrade/
    # Most notable upgrade is backends: https://emcee.readthedocs.io/en/stable/tutorials/monitor/
    # Have some useful implementation in old_run_driver.py, incl for schwimmbad.
    

    # Initialize a generator to provide the data. They changed the arg
    # storechain -> store sometime between v2.2.1 (iorek) and v3.0rc2 (cluster)
    from emcee import __version__ as emcee_version
    # iorek is on v2, cluster and kazul are v3
    if emcee_version[0] == '2':     
        # Initialize the sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        args=(run_path, param_info, mol),
                                        pool=pool)
        run = sampler.sample(pos, iterations=nsteps, storechain=False)
        
        # No backend here, so gotta do it manually.
        lnprobs = []    
        for i, result in enumerate(run):
            pos, lnprobs, blob = result

            # Log out the new positions
            with open(chain_filename, 'a') as f:
                new_step = [np.append(pos[k], lnprobs[k]) for k in range(nwalkers)]

                from datetime import datetime
                now = datetime.now().strftime('%H:%M, %m/%d')

                print("[{}] Adding a new step to the chain".format(now))
                np.savetxt(f, new_step, delimiter=',')
                
    else:  # for cluster and kazul
        
        # Can now tell walkers to move in different (not just stretch) ways
        # https://emcee.readthedocs.io/en/stable/user/moves/#moves-user
        # TODO: Look intio using other moves.
        move = emcee.moves.StretchMove
        
        # There is also now a default backend builtin
        filename = "tutorial.h5"  #TODO: Update this
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        args=(run_path, param_info, mol),
                                        pool=pool, moves=move,
                                        backend=backend)
        # Note that nsteps should be huge, since ideally we converge before hitting it.
        run = sampler.sample(pos, iterations=nsteps, progress=True)
        
        
        # Pulled from https://emcee.readthedocs.io/en/stable/tutorials/monitor/
        # index = 0
        # autocorr = np.empty(nsteps)
        autocorr = []
        old_tau = np.inf

        for sample in run:
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            # autocorr[index] = np.mean(tau)
            autocorr.append(np.mean(tau))
            # index += 1

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau


    print("Ended run")



    
    
    
    

def main():
    """Establish and evaluate some custom argument options.

    This is called when we run the whole thing: run_driver.py -r does a run.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''Blah''')

    # TODO: Allow user to provide run name to resume here. Not action=store_true
    parser.add_argument('-r', '--resume', help='Want to resume a run?')
    parser.add_argument('-hco', '--run_hco', action='store_true',
                        help='Begin an HCO+ run.')
    parser.add_argument('-hcn', '--run_hcn', action='store_true',
                        help='Begin an HCN run.')
    parser.add_argument('-co', '--run_co', action='store_true',
                        help='Begin a CO run.')
    parser.add_argument('-cs', '--run_cs', action='store_true',
                        help='Begin a CS run.')
    parser.add_argument('-multi', '--run_multi', action='store_true',
                        help='Begin a multi-line (HCO+/HCN) run.')

    args = parser.parse_args()



    if args.run_hco or args.run_hcn or args.run_co or args.run_cs or args.run_multi:
        # Set up the parallelization
         # Iorek boxes is on v2, cluster and kazul are v3
        from emcee import __version__ as emcee_version
        if emcee_version[0] == '2':
            from emcee.utils import MPIPool
            pool = MPIPool()

            # Tell the difference between master and worker processes
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
        else:
            # from schwimmbad import MultiPool
            from schwimmbad import MPIPool
            pool = MultiPool(args.n_cores)


        # Do the thing
        resume_run = None
        if args.resume:
            resume_run = args.resume
            
        if args.run_hco:
            run_emcee(mol='hco', lnprob=lnprob, pool=pool, resume_run=resume_run)

        if args.run_hcn:
            run_emcee(mol='hcn', lnprob=lnprob, pool=pool, resume_run=resume_run)

        if args.run_co:
            run_emcee(mol='co', lnprob=lnprob, pool=pool, resume_run=resume_run)

        if args.run_cs:
            run_emcee(mol='cs', lnprob=lnprob, pool=pool, resume_run=resume_run)

        if args.run_multi:
            run_emcee(mol='multi', lnprob=lnprob, pool=pool, resume_run=resume_run)


        pool.close()





if __name__ == '__main__':
    main()







# The End
