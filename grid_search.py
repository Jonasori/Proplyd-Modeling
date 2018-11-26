"""Run a grid search."""


import csv
import time
import itertools
import numpy as np
import pandas as pd
import cPickle as pickle
import subprocess as sp


# Local package files
from utils import makeModel, sumDisks, chiSq
from tools import icr, sample_model_in_uvplane, already_exists, remove
from analysis import plot_gridSearch_log, plot_step_duration, plot_fits
from constants import mol, today, dataPath
from run_params import diskAParams, diskBParams


# A little silly, but an easy way to name disks by their disk index (DI)
dnames = ['A', 'B']
# Set up the header of a list to keep track of how long each iteration takes.
times = [['step', 'duration']]

# An up-to-date list of the params being queried.
param_names = ['v_turb', 'zq', 'r_crit', 'rho_p', 't_mid', 'PA', 'incl',
               'pos_x', 'pos_y', 'v_sys', 't_atms', 't_atms',
               'r_out', 'm_disk', 'x_mol']

shape = [len(diskBParams[p]) for p in param_names]

# Prep some storage space for all the chisq vals
diskARawX2 = np.zeros((len(diskAParams[0]), len(diskAParams[1]),
                       len(diskAParams[2]), len(diskAParams[3]),
                       len(diskAParams[4]), len(diskAParams[5]),
                       len(diskAParams[6]), len(diskAParams[7]),
                       len(diskAParams[8]), len(diskAParams[9])
                       ))

diskBRawX2 = np.zeros((len(diskBParams[0]), len(diskBParams[1]),
                       len(diskBParams[2]), len(diskBParams[3]),
                       len(diskBParams[4]), len(diskBParams[5]),
                       len(diskBParams[6]), len(diskBParams[7]),
                       len(diskBParams[8]), len(diskBParams[9])
                       ))

diskARedX2 = np.zeros((len(diskAParams[0]), len(diskAParams[1]),
                       len(diskAParams[2]), len(diskAParams[3]),
                       len(diskAParams[4]), len(diskAParams[5]),
                       len(diskAParams[6]), len(diskAParams[7]),
                       len(diskAParams[8]), len(diskAParams[9])
                       ))

diskBRedX2 = np.zeros((len(diskBParams[0]), len(diskBParams[1]),
                       len(diskBParams[2]), len(diskBParams[3]),
                       len(diskBParams[4]), len(diskBParams[5]),
                       len(diskBParams[6]), len(diskBParams[7]),
                       len(diskBParams[8]), len(diskBParams[9])
                       ))


# GRID SEARCH OVER ONE DISK HOLDING OTHER CONSTANT
def gridSearch(VariedDiskParams,
               StaticDiskParams,
               DI,
               modelPath,
               num_iters, steps_so_far=1,
               cut_central_chans=False):
    """
    Run a grid search over parameter space.

    Args:
        VariedDiskParams (list of lists): lists of param vals to try.
        StaticDiskParams (list of floats) Single vals for the static model.
        DI: Disk Index of varied disk (0 or 1).
            If 0, A is the varied disk and vice versa
    Returns: 	[X2 min value, Coordinates of X2 min]
    Creates:	Best fit two-disk model
    """
    # Disk names should be the same as the output from makeModel()?

    # Initiate a list to hold the rows of the df
    df_rows = []

    # Get the index of the static disk, name the outputs
    DIs = abs(DI - 1)
    outNameVaried = modelPath + 'fitted_' + dnames[DI]
    outNameStatic = modelPath + 'static_' + dnames[DIs]

    makeModel(StaticDiskParams, outNameStatic, DIs)

    # Set up huge initial chi squared values so that they can be improved upon.
    minRedX2 = 1e10
    minX2Vals = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    counter = steps_so_far

    # Pull the params we're looping over.
    # All these are np.arrays (sometimes of length 1)
    all_v_turb  = params['v_turb']
    all_zq      = params['zq']
    all_r_crit  = params['r_crit']
    all_rho_p   = params['rho_p']
    all_t_mid   = params['t_mid']
    all_PA      = params['PA']
    all_incl    = params['incl']
    all_pos_x   = params['pos_x']
    all_pos_y   = params['pos_y']
    all_v_sys   = params['v_sys']
    all_t_atms  = params['t_atms']
    all_t_atms  = params['t_atms']
    all_r_out   = params['r_out']
    all_m_disk  = params['m_disk']
    all_x_mol   = params['x_mol']

    """ Grids by hand
        for i in range(0, len(Tatms)):
            for j in range(0, len(Tqq)):
                for l in range(0, len(R_out)):
                    for k in range(0, len(Xmol)):
                        for m in range(0, len(PA)):
                            for n in range(0, len(Incl)):
                                for o in range(0, len(Pos_X)):
                                    for p in range(0, len(Pos_Y)):
                                        for q in range(0, len(V_sys)):
                                            for r in range(0, len(M_disk)):
                                                # Create a list of floats to feed makeModel()
                                                """

    # I think that itertools.product does the same thing as the nested loops above
    # Loop over everything, even though only most params aren't varied.
    ps = itertools.product(range(len(all_v_turb)), range(len(all_zq)),
                           range(len(all_r_crit)), range(len(all_rho_p)),
                           range(len(all_t_mid)),  range(len(all_PA)),
                           range(len(all_incl)),   range(len(all_pos_x)),
                           range(len(all_pos_y)),  range(len(all_all_v_sys)),
                           range(len(all_t_atms)), range(len(all_t_qq)),
                           range(len(all_r_out)),  range(len(all_m_disk)),
                           range(len(all_x_mol)))
    # Pull floats out of those lists.
    for i, j, k, l, m, n, o, p, q, r, s, t, u, v, w in ps:
        begin = time.time()
        v_turb = all_v_turb[i]
        zq = all_zq[j]
        r_crit = all_r_crit[k]
        rho_p = all_rho_p[l]
        t_mid = all_t_mid[m]
        PA = all_PA[n]
        incl = all_incl[o]
        pos_x = all_pos_x[p]
        pos_y = all_pos_y[q]
        v_sys = all_all_v_sys[r]
        t_atms = all_t_atms[s]
        t_qq = all_t_qq[t]
        r_out = all_r_out[u]
        m_disk = all_m_disk[v]
        x_mol = all_x_mol[w]

        params = {'v_turb': v_turb,
                  'zq': zq,
                  'r_crit': r_crit,
                  'rho_p': rho_p
                  't_mid': t_mid,
                  'PA': PA,
                  'incl': incl,
                  'pos_x': pos_x,
                  'pos_y': pos_y,
                  'v_sys': v_sys,
                  't_atms': t_atms,
                  't_qq': t_qq,
                  'r_out': r_out,
                  'm_disk': m_disk,
                  'x_mol': x_mol
                  }


        # params = [zq, r_crit, rho_p, t_mid, PA, incl, pos_x, pos_y, v_sys,
        #           t_atms, t_qq, r_out, m_disk, x_mol]

        # Things left to fix:
        # - Chi2 grids (up top)
        # - Chi2 grid population (below)
        # - df write out (maybe have it write out every step while we're at it)
        # - printing stuff
        # - param read-in in makeModel()

        # Print out some info
        print "\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        print "Currently fitting for: " + outNameVaried
        print "Beginning model " + str(counter) + "/" + str(num_iters)
        print "Fit Params:"
        for param in parans:
            print param, params[param]
        # This isn't really necessary to have
        print "\nStatic params:"
        for static in StaticDiskParams:
            print static, StaticDiskParams[static]

        # Make a new disk, sum them, sample in vis-space.
        makeModel(params, outNameVaried, DI)
        sumDisks(outNameVaried, outNameStatic, modelPath)
        sample_model_in_uvplane(modelPath, mol=mol)

        # Visibility-domain chi-squared evaluation
        rawX2, redX2 = chiSq(modelPath, cut_central_chans=cut_central_chans)

        # It's ok to split these up by disk since disk B's
        # best params are independent of where disk A is.
        if DI == 0:
            diskARawX2[i, j, k, l, m, n, o, p, q, r, s, t, u, v, w] = rawX2
            diskARedX2[i, j, k, l, m, n, o, p, q, r, s, t, u, v, w] = redX2
        else:
            diskBRawX2[i, j, k, l, m, n, o, p, q, r, s, t, u, v, w] = rawX2
            diskBRedX2[i, j, k, l, m, n, o, p, q, r, s, t, u, v, w] = redX2

        print "\n\n"
        print "Raw Chi-Squared value:	 ", rawX2
        print "Reduced Chi-Squared value:", redX2

        df_row = {'V Turb': v_turb,
                  'Zq': zq,
                  'R crit': r_crit,
                  'Density Str': rho_p
                  'T mid': t_mid,
                  'PA': PA,
                  'Incl': incl,
                  'Pos x': pos_x,
                  'Pos Y': pos_y,
                  'V Sys': v_sys,
                  'T atms': t_atms,
                  'Temp Str': t_qq,
                  'Outer Radius': r_out,
                  'Disk Mass': m_disk,
                  'Molecular Abundance': x_mol
                  'Raw Chi2': rawX2,
                  'Reduced Chi2': redX2,
                  }
        df_rows.append(df_row)
        # Maybe want to re-export the df every time here?

        if redX2 > 0 and redX2 < minRedX2:
            minRedX2 = redX2
            minX2Vals = [ta, tqq, xmol, r_out, pa, incl, pos_x, pos_y, vsys, m_disk]
            sp.call(
                'mv {}.fits {}_bestFit.fits'.format(modelPath, modelPath),
                shell=True)
            print "Best fit happened; moved file"

        # Now clear out all the files (im, vis, uvf, fits)
        remove(modelPath + ".*")
        # sp.call('rm -rf {}.*'.format(modelPath),
        #         shell=True)

        # Loop this.
        print "Min. Chi-Squared value so far:", minRedX2

        counter += 1
        finish = time.time()
        times.append([counter, finish - begin])


    # Finally, make the best-fit model for this disk
    makeModel(minX2Vals, outNameVaried, DI)
    print "Best-fit model for disk", dnames[DI], " created: ", modelPath, ".fits\n\n"

    # Knit the dataframe
    step_log = pd.DataFrame(df_rows)
    print "Shape of long-log data frame is ", step_log.shape

    # Give the min value and where that value is
    print "Minimum Chi2 value and where it happened: ", [minRedX2, minX2Vals]
    return step_log


# PERFORM A FULL RUN USING FUNCTIONS ABOVE #
def fullRun(diskAParams, diskBParams,
            use_a_previous_result=False,
            cut_central_chans=False):
    """Run it all.

    diskXParams are fed in from full_run.py,
    where the parameter selections are made.
    """
    t0 = time.time()

    # Calculate the number of steps and consequent runtime
    na = 1
    for a in range(0, len(diskAParams)):
        na *= len(diskAParams[a])

    nb = 1
    for b in range(0, len(diskBParams)):
        nb *= len(diskBParams[b])

    n, dt = na + nb, 2.1
    t = n * dt
    if t <= 60:
        t = str(round(n * dt, 2)) + " minutes."
    elif t > 60 and t <= 1440:
        t = str(round(n * dt/60, 2)) + " hours."
    elif t >= 1440:
        t = str(round(n * dt/1440, 2)) + " days."


    # Begin setting up symlink and get directory paths lined up
    this_run_basename = today + '_' + mol
    this_run = this_run_basename
    modelPath = './gridsearch_runs/' + this_run + '/' + this_run
    run_counter = 2
    # while already_exists_old(modelPath) is True:
    while already_exists(modelPath) is True:
        this_run = this_run_basename + '-' + str(run_counter)
        modelPath = './gridsearch_runs/' + this_run + '/' + this_run

        run_counter += 1

    # Parameter Check:
    print "\nThis run will fit for", mol.upper()
    print "It will iterate through these parameters for Disk A:"
    for p in range(len(diskAParams)):
        print param_names[p], ': ', diskAParams[p]
    print "\nAnd these values for Disk B:"
    for p in range(len(diskBParams)):
        print param_names[p], ': ', diskBParams[p]


    print "\nThis run will take", n, "steps, spanning about", t
    print "Output will be in", modelPath, '\n'
    response = raw_input('Sound good? (Enter to begin, anything else to stop)\n')
    if response != "":
        return "\nGo fix whatever you don't like and try again.\n\n"
    else:
        print "Sounds good!\n"

    new_dir = '/Volumes/disks/jonas/modeling/gridsearch_runs/' + this_run
    sp.call(['mkdir', 'gridsearch_runs/' + this_run])

    # CHECK FOR REUSE
    """This is a little bit janky looking but makes sense. Since we are
    treating the two disks as independent, then if, in one run, we find good
    fits (no edge values), then it doesn't make sense to run that grid again;
    it would be better to just grab the relevant information from that run
    and only fit the disk that needs fitting. That's what this is for."""
    to_skip = ''
    if use_a_previous_result is True:
        response2 = raw_input(
            'Please enter the path to the .fits file to use from a previous',
            'run (should be ./models/date/run_date/datefitted_[A/B].fits)\n')
        if 'A' in response2:
            to_skip = 'fitted_A'
        elif 'B' in response2:
            to_skip = 'fitted_B'
        else:
            print "Bad path; must have 'fitted_A or fitted_B' in it. Try again"
            return




    # STARTING THE RUN #
    # Make the initial static model (B), just with the first parameter values
    dBInit = [i[0] for i in diskBParams]

    # Grid search over Disk A, retrieve the resulting pd.DataFrame
    if to_skip != 'A':
        df_A_fit = gridSearch(diskAParams, dBInit, 0, modelPath, n,
                              cut_central_chans=cut_central_chans)

    # Find where the chi2 is minimized and save it
    idx_of_BF_A = df_A_fit.index[df_A_fit['Reduced Chi2'] == np.min(
        df_A_fit['Reduced Chi2'])][0]
    print "Index of Best Fit, A is ", idx_of_BF_A

    # Make a list of those parameters to pass the next round of grid searching.
    Ps_A = [df_A_fit[param][idx_of_BF_A] for param in  df_A_fit.columns]
    fit_A_params = np.array(Ps_A)

    print "First disk has been fit\n"

    # Now search over the other disk
    df_B_fit = gridSearch(diskBParams, fit_A_params, 1, modelPath,
                          n, steps_so_far=na,
                          cut_central_chans=cut_central_chans)

    idx_of_BF_B = df_B_fit.index[df_B_fit['Reduced Chi2'] == np.min(
        df_B_fit['Reduced Chi2'])][0]

    Ps_B = [df_B_fit[param][idx_of_BF_B] for param in  df_B_fit.columns]
    fit_B_params = np.array(Ps_B)


    # Bind the data frames, output them.
    # Reiterated in tools.py/depickler(), but we can unwrap these vals with:
    # full_log.loc['A', :] to get all the columns for disk A, or
    # full_log[:, 'Incl.'] to see which inclinations both disks tried.
    full_log = pd.concat([df_A_fit, df_B_fit], keys=['A', 'B'], names=['Disk'])
    # Pickle the step log df.
    pickle.dump(full_log, open('{}_step-log.pickle'.format(modelPath), "wb"))
    # To read the pickle:
    # f = pickle.load(open('{}_step-log.pickle'.format(modelPath), "rb"))

    # Finally, Create the final best-fit model and residuals
    print "\n\nCreating best fit model now"
    sample_model_in_uvplane(modelPath + '_bestFit', mol=mol)
    sample_model_in_uvplane(modelPath + '_bestFit', options='subtract', mol=mol)
    icr(modelPath + '_bestFit', mol=mol)
    icr(modelPath + '_bestFit_resid', mol=mol)
    print "Best-fit model created: " + modelPath + "_bestFit.im\n\n"

    # Calculate and present the final X2 values.
    finalX2s = chiSq(modelPath + '_bestFit')
    print "Final Raw Chi-Squared Value: ", finalX2s[0]
    print "Final Reduced Chi-Squared Value: ", finalX2s[1]

    # Clock out
    t1 = time.time()
    t_total = (t1 - t0)/60
    # n+4 to account for best-fit model making and static disks in grid search
    t_per = str(t_total/(n + 4))

    with open(modelPath + '_stepDurations.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerows(times)

    print "\n\nFinal run duration was", t_total/60, ' hours'
    print 'with each step taking on average', t_per, ' minutes'

    # log file w/ best fit vals, range queried, indices of best vals, best chi2
    with open(modelPath + '_summary.log', 'w') as f:
        s0 = '\nLOG FOR RUN ON' + today + ' FOR THE ' + mol + ' LINE'
        s1 = '\nBest Chi-Squared values [raw, reduced]:\n' + str(finalX2s)
        s2 = '\n\n\nParameter ranges queried:\n'
        s3 = '\nDisk A:\n'
        for i, ps in enumerate(diskAParams):
            s3 = s3 + param_names[i] + str(ps) + '\n'
        s4 = '\nDisk B:\n'
        for i, ps in enumerate(diskBParams):
            s4 = s4 + param_names[i] + str(ps) + '\n'
        s5 = '\n\n\nBest-fit values (Tatm, Tqq, Xmol, outerR, PA, Incl):'
        s6 = '\nDisk A:\n' + str(fit_A_params)
        s7 = '\nDisk B:\n' + str(fit_B_params)
        s8 = '\n\n\nFinal run duration was' + str(t_total/60) + 'hours'
        s9 = '\nwith each step taking on average' + t_per + 'minutes'
        s10 = '\n\nData file used was ' + dataPath
        s = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10
        f.write(s)

    plot_gridSearch_log(modelPath, show=False)
    plot_step_duration(modelPath, show=False)
    plot_fits(modelPath + '_bestFit.fits', show=False)
    print "Successfully finished everything."

# The End
