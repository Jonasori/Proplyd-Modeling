"""Run a grid search.

The only part of this that changes for the four-line fit is in the model making
itself, deep in the grid search.

It's gonna be really hard to fit for some parameters individually in a grid search without a significant rewrite. Right?




- Common temperature structure (may have to vary zq?  try fixed first? *Would varying this happen line-by-line?)
- Vary abundance and radius for each molecule separately
- Keep disk geometry the same for all molecules
- Sum raw chi^2 for each of four molecules

If you don't get a reasonable solution, check what Sam did, but I think maybe we wound up having to vary T_atm separately for each molecule?  Important to try with a consistent temperature structure first so that we can find out how badly it does.

Also check Sam's paper to see if there are descriptions of number density thresholds for upper/lower molecular boundaries, based on photodissociation cross-sections and/or freeze-out.
"""


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
from constants import today, dataPath
# from run_params import diskAParams_fourline, diskBParams_fourline


# A little silly, but an easy way to name disks by their disk index (DI)
dnames = ['A', 'B']
# Set up the header of a list to keep track of how long each iteration takes.
times = [['step', 'duration']]

mols = ['hco', 'hcn', 'cs', 'co']

# An up_to_date list of the params being queried.
param_names = diskAParams_fourline.keys()
# Prep some storage space for all the chisq vals
diskA_shape = [len(diskAParams_fourline[p]) for p in param_names]
diskB_shape = [len(diskBParams_fourline[p]) for p in param_names]

diskARawX2, diskARedX2 = np.zeros(diskA_shape), np.zeros(diskA_shape)
diskBRawX2, diskBRedX2 = np.zeros(diskB_shape), np.zeros(diskB_shape)


def gridSearch(VariedDiskParams,
               StaticDiskParams,
               DI, modelPath,
               num_iters, steps_so_far=1,
               cut_central_chans=False):
    """
    Run a grid search over parameter space.

    Args:
        VariedDiskParams (list of lists): lists of param vals to try.
        StaticDiskParams (list of floats) Single vals for the static model.
        DI: Disk Index of varied disk (0 or 1).
            If 0, A is the varied disk and vice versa
    Returns: 	data-frame log of all the steps
    Creates:	Best fit two-disk model
    """
    counter = steps_so_far

    # Initiate a list to hold the rows of the df
    df_rows = []

    # Get the index of the static disk, name the outputs
    DIs = abs(DI - 1)
    outNameVaried = modelPath + '-fitted_' + dnames[DI]
    outNameStatic = modelPath + '-static_' + dnames[DIs]

    # Make four static disks
    print "Making static disks"
    for mol in mols:
        makeModel(StaticDiskParams[mol], outNameStatic + '_' + mol, DIs, mol)
    print "Finished making static disks"

    ### FOUR LINES
    # Make a new disk, sum them, sample in vis-space.
    # Not sure why this is here.
    """
    makeModel(params_hco, outNameVaried + '_hco', DI, mol)
    makeModel(params_hcn, outNameVaried + '_hcn', DI, mol)
    makeModel(params_co, outNameVaried + '_co', DI, mol)
    makeModel(params_cs, outNameVaried + '_cs', DI, mol)
    """






    # Set up huge initial chi squared values so that they can be improved upon.
    minRedX2 = 1e10
    # Initiate a best-fit param dict
    minX2Vals = {'co': {}, 'cs': {}, 'hco': {}, 'hcn': {}}
    for mol in mols:
        for p in VariedDiskParams:
            # Elements of paramdicts are sometimes arrays and sometimes dicts
            if type(VariedDiskParams[p]) == dict:
                minX2Vals[mol][p] = VariedDiskParams[p][mol][0]
            else:
                minX2Vals[mol][p] = VariedDiskParams[p][0]

    # Pull the params we're looping over.
    # All these are np.arrays (sometimes of length 1)
    all_v_turb  = VariedDiskParams['v_turb']
    all_r_crit  = VariedDiskParams['r_crit']
    all_rho_p   = VariedDiskParams['rho_p']
    all_t_mid   = VariedDiskParams['t_mid']
    all_PA      = VariedDiskParams['PA']
    all_incl    = VariedDiskParams['incl']
    all_pos_x   = VariedDiskParams['pos_x']
    all_pos_y   = VariedDiskParams['pos_y']
    all_v_sys   = VariedDiskParams['v_sys']
    all_t_qq    = VariedDiskParams['t_qq']
    all_zq      = VariedDiskParams['zq']

    all_t_atms_co  = VariedDiskParams['t_atms']['co']
    all_t_atms_cs  = VariedDiskParams['t_atms']['cs']
    all_t_atms_hco  = VariedDiskParams['t_atms']['hco']
    all_t_atms_hcn  = VariedDiskParams['t_atms']['hcn']

    all_r_out_co   = VariedDiskParams['r_out']['co']
    all_r_out_cs   = VariedDiskParams['r_out']['cs']
    all_r_out_hco   = VariedDiskParams['r_out']['hco']
    all_r_out_hcn   = VariedDiskParams['r_out']['hcn']

    all_m_disk_co  = VariedDiskParams['m_disk']['co']
    all_m_disk_cs  = VariedDiskParams['m_disk']['cs']
    all_m_disk_hco  = VariedDiskParams['m_disk']['hco']
    all_m_disk_hcn  = VariedDiskParams['m_disk']['hcn']

    all_x_mol_co   = VariedDiskParams['x_mol']['co']
    all_x_mol_cs   = VariedDiskParams['x_mol']['cs']
    all_x_mol_hco   = VariedDiskParams['x_mol']['hco']
    all_x_mol_hcn   = VariedDiskParams['x_mol']['hcn']


    # This is horrendous good Lord.
    ps = itertools.product(range(len(VariedDiskParams.keys()[0])),
                           range(len(VariedDiskParams.keys()[1])),
                           range(len(VariedDiskParams.keys()[2])),
                           range(len(VariedDiskParams.keys()[3])),
                           range(len(VariedDiskParams.keys()[4])),
                           range(len(VariedDiskParams.keys()[5])),
                           range(len(VariedDiskParams.keys()[6])),
                           range(len(VariedDiskParams.keys()[7])),
                           range(len(VariedDiskParams.keys()[8])),
                           range(len(VariedDiskParams.keys()[9])),
                           range(len(VariedDiskParams.keys()[10])),
                           range(len(VariedDiskParams.keys()[11])),
                           range(len(VariedDiskParams.keys()[12])),
                           range(len(VariedDiskParams.keys()[13])),
                           range(len(VariedDiskParams.keys()[14])),
                           range(len(VariedDiskParams.keys()[15])),
                           range(len(VariedDiskParams.keys()[16])),
                           range(len(VariedDiskParams.keys()[17])),
                           range(len(VariedDiskParams.keys()[18])),
                           range(len(VariedDiskParams.keys()[19])),
                           range(len(VariedDiskParams.keys()[20])),
                           range(len(VariedDiskParams.keys()[21])),
                           range(len(VariedDiskParams.keys()[22])),
                           range(len(VariedDiskParams.keys()[23])),
                           range(len(VariedDiskParams.keys()[24])),
                           range(len(VariedDiskParams.keys()[25])),
                           range(len(VariedDiskParams.keys()[26])),
                           range(len(VariedDiskParams.keys()[27])),
                           range(len(VariedDiskParams.keys()[28])),
                           range(len(VariedDiskParams.keys()[29])),
                           range(len(VariedDiskParams.keys()[30]))
                           )

    # Pull floats out of those lists.
    for a, b, c, d, e, f, g, h, i, j, rco, rcs, rhco, rhcn, tco, tcs, thco, thcn, xco, xcs, xhco, xhcn, mco, mcs, mhco, mhcn in ps:
        begin = time.time()
        v_turb  = all_v_turb[a]
        r_crit  = all_r_crit[b]
        rho_p   = all_rho_p[c]
        t_mid   = all_t_mid[d]
        PA      = all_PA[e]
        incl    = all_incl[f]
        pos_x   = all_pos_x[g]
        pos_y   = all_pos_y[h]
        v_sys   = all_v_sys[i]
        t_qq    = all_t_qq[j]
        zq      = all_zq[k]

        r_out_co = all_r_out_co[rco]
        r_out_cs = all_r_out_cs[rcs]
        r_out_hco = all_r_out_hco[rhco]
        r_out_hcn = all_r_out_hcn[rhcn]

        t_atms_co = all_t_atms_co[tco]
        t_atms_cs = all_t_atms_cs[tcs]
        t_atms_hco = all_t_atms_hco[thco]
        t_atms_hcn = all_t_atms_hcn[thcn]

        x_mol_co = all_x_mol_co[xco]
        x_mol_cs = all_x_mol_cs[xcs]
        x_mol_hco = all_x_mol_hco[xhco]
        x_mol_hcn = all_x_mol_hcn[xhcn]

        m_disk_co = all_m_disk_co[mco]
        m_disk_cs = all_m_disk_cs[mcs]
        m_disk_hco = all_m_disk_hco[mhco]
        m_disk_hcn = all_m_disk_hcn[mhcn]


        # Print out some info
        print "\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        print "Currently fitting for: " + outNameVaried
        print "Beginning model " + str(counter) + "/" + str(num_iters)


        # Bundle them up into lines
        params_hco = {'v_turb': v_turb,
                      'r_crit': r_crit,
                      'rho_p': rho_p,
                      't_mid': t_mid,
                      'PA': PA,
                      'incl': incl,
                      'pos_x': pos_x,
                      'pos_y': pos_y,
                      'v_sys': v_sys,
                      't_qq': t_qq,
                      'zq': zq,
                      'r_out': r_out_hco,
                      'x_mol': x_mol_hco,
                      't_atms': t_atms_hco,
                      'm_disk': m_disk_hco
                      }
        params_hcn = {'v_turb': v_turb,
                      'r_crit': r_crit,
                      'rho_p': rho_p,
                      't_mid': t_mid,
                      'PA': PA,
                      'incl': incl,
                      'pos_x': pos_x,
                      'pos_y': pos_y,
                      'v_sys': v_sys,
                      't_qq': t_qq,
                      'zq': zq,
                      'r_out': r_out_hcn,
                      't_atms': t_atms_hcn,
                      'x_mol': x_mol_hcn,
                      'm_disk': m_disk_hcn
                      }
        params_co = {'v_turb': v_turb,
                      'r_crit': r_crit,
                      'rho_p': rho_p,
                      't_mid': t_mid,
                      'PA': PA,
                      'incl': incl,
                      'pos_x': pos_x,
                      'pos_y': pos_y,
                      'v_sys': v_sys,
                      't_qq': t_qq,
                      'zq': zq,
                      'r_out': r_out_co,
                      't_atms': t_atms_co,
                      'x_mol': x_mol_co,
                      'm_disk': m_disk_co
                      }
        params_cs = {'v_turb': v_turb,
                      'r_crit': r_crit,
                      'rho_p': rho_p,
                      't_mid': t_mid,
                      'PA': PA,
                      'incl': incl,
                      'pos_x': pos_x,
                      'pos_y': pos_y,
                      'v_sys': v_sys,
                      't_qq': t_qq,
                      'zq': zq,
                      'r_out': r_out_cs,
                      't_atms': t_atms_cs,
                      'x_mol': x_mol_cs,
                      'm_disk': m_disk_cs
                      }


        ### FOUR LINES
        # Make a new disk, sum them, sample in vis-space.
        makeModel(params_hco, outNameVaried + '_hco', DI, mol)
        makeModel(params_hcn, outNameVaried + '_hcn', DI, mol)
        makeModel(params_co, outNameVaried + '_co', DI, mol)
        makeModel(params_cs, outNameVaried + '_cs', DI, mol)

        rawX2, redX2 = 0, 0
        for mol in mols:
            sumDisks(outNameVaried + '_' + mol, outNameStatic + '_' + mol,
                     modelPath, mol)
            sample_model_in_uvplane(modelPath, mol)

            # Visibility-domain chi-squared evaluation
            X2s = chiSq(modelPath, mol, cut_central_chans=cut_central_chans)
            rawX2 += X2s[0]
            redX2 += X2s[1]






        # It's ok to split these up by disk since disk B's
        # best params are independent of where disk A is.
        if DI == 0:
            diskARawX2[a, b, c, d, e, f, g, h, i, j,
                       rco, rcs, rhco, rhcn, tco, tcs, thco, thcn,
                       xco, xcs, xhco, xhcn, mco, mcs, mhco, mhcn] = rawX2
            diskARedX2[a, b, c, d, e, f, g, h, i, j,
                       rco, rcs, rhco, rhcn, tco, tcs, thco, thcn,
                       xco, xcs, xhco, xhcn, mco, mcs, mhco, mhcn] = redX2
        else:
            diskBRawX2[a, b, c, d, e, f, g, h, i, j,
                       rco, rcs, rhco, rhcn, tco, tcs, thco, thcn,
                       xco, xcs, xhco, xhcn, mco, mcs, mhco, mhcn] = rawX2
            diskBRedX2[a, b, c, d, e, f, g, h, i, j,
                       rco, rcs, rhco, rhcn, tco, tcs, thco, thcn,
                       xco, xcs, xhco, xhcn, mco, mcs, mhco, mhcn] = redX2

        print "\n\n"
        print "Raw Chi-Squared value:	 ", rawX2
        print "Reduced Chi-Squared value:", redX2

        df_row = params
        df_row['Raw Chi2'] = rawX2
        df_row['Reduced Chi2'] = redX2
        df_rows.append(df_row)
        # Maybe want to re-export the df every time here?

        # If this is the best fit so far, log it as such
        if redX2 > 0 and redX2 < minRedX2:
            minRedX2 = redX2
            minX2Vals = params
            for mol in mols:
                sp.call(
                    'mv {}.fits {}_bestFit-' + mol + '.fits'.format(modelPath, modelPath),
                    shell=True)
                print "Best fit happened; moved file"

        # Now clear out all the files (im, vis, uvf, fits)
        remove(modelPath + ".*")

        print "Min. Chi-Squared value so far:", minRedX2

        counter += 1
        finish = time.time()
        times.append([counter, finish - begin])


    # Finally, make the best-fit model for this disk
    # I don't think this is actually necessary
    # [makeModel(minX2Vals, outNameVaried + '_' + mol, DI, mol) for mol in mols]

    print "Best-fit model for disk", dnames[DI], " created: ", modelPath, ".fits\n\n"

    # Knit the dataframe
    step_log = pd.DataFrame(df_rows)
    print "Shape of long-log data frame is ", step_log.shape

    # Give the min value and where that value is
    print "Minimum Chi2 value and where it happened: ", [minRedX2, minX2Vals]
    return step_log


# PERFORM A FULL RUN USING FUNCTIONS ABOVE #
def fullRun(diskAParams_fourline, diskBParams_fourline,
            use_a_previous_result=False,
            cut_central_chans=False):
    """Run it all.

    diskXParams are fed in from full_run.py,
    where the parameter selections are made.
    """
    t0 = time.time()

    # Calculate the number of steps and consequent runtime
    na = 1
    for a in diskAParams_fourline:
        na *= len(diskAParams_fourline[a])

    nb = 1
    for b in diskBParams_fourline:
        nb *= len(diskBParams_fourline[b])

    n, dt = na + nb, 10.
    t = n * dt
    if t <= 60:
        t = str(round(n * dt, 2)) + " minutes."
    elif t > 60 and t <= 1440:
        t = str(round(n * dt/60, 2)) + " hours."
    elif t >= 1440:
        t = str(round(n * dt/1440, 2)) + " days."


    # Begin setting up symlink and get directory paths lined up
    this_run_basename = today + '_four-line-fit'
    this_run = this_run_basename
    modelPath = './gridsearch_runs/' + this_run + '/' + this_run
    run_counter = 2
    # while already_exists_old(modelPath) is True:
    while already_exists(modelPath) is True:
        this_run = this_run_basename + '-' + str(run_counter)
        modelPath = './gridsearch_runs/' + this_run + '/' + this_run
        run_counter += 1

    # Parameter Check:
    print "\nThis run will fit for all four lines simultaneously"
    print "It will iterate through these parameters for Disk A:"
    for p in diskAParams_fourline:
        print p, ': ', diskAParams_fourline[p]
    print "\nAnd these values for Disk B:"
    for p in diskBParams_fourline:
        print p, ': ', diskBParams_fourline[p]


    print "\nThis run will take", n, "steps, spanning about", t
    print "Output will be in", modelPath, '\n'
    response = raw_input('Sound good? (Enter to begin, anything else to stop)\n')
    if response != "":
        return "\nGo fix whatever you don't like and try again.\n\n"
    else:
        print "Sounds good!\n"

    new_dir = '/Volumes/disks/jonas/modeling/gridsearch_runs/' + this_run
    sp.call(['mkdir', 'gridsearch_runs/' + this_run])


    # STARTING THE RUN #
    # Make the initial static model (B), just with the first parameter values
    dBInit = {'co': {}, 'cs': {}, 'hco': {}, 'hcn': {}}
    for mol in mols:
        for p in diskBParams_fourline:
            # Since some of the values are line-specific, stored in dicts, dig those out.
            if type(diskBParams_fourline[p]) == dict:
                dBInit[mol][p] = diskBParams_fourline[p][mol][0]
            else:
                dBInit[mol][p] = diskBParams_fourline[p][0]


    # Grid search over Disk A, retrieve the resulting pd.DataFrame
    df_A_fit = gridSearch(diskAParams_fourline, dBInit, 0,
                          modelPath, n,
                          cut_central_chans=cut_central_chans)

    # Find where the chi2 is minimized and save it
    idx_of_BF_A = df_A_fit.index[df_A_fit['Reduced Chi2'] == np.min(
        df_A_fit['Reduced Chi2'])][0]
    print "Index of Best Fit, A is ", idx_of_BF_A

    # Make a list of those parameters to pass the next round of grid searching.
    fit_A_params = {}
    for param in df_A_fit.columns:
        fit_A_params[param] = df_A_fit[param][idx_of_BF_A]

    print "First disk has been fit\n"

    # Now search over the other disk
    df_B_fit = gridSearch(diskBParams_fourline, fit_A_params,
                          1, modelPath,
                          n, steps_so_far=na,
                          cut_central_chans=cut_central_chans)

    idx_of_BF_B = df_B_fit.index[df_B_fit['Reduced Chi2'] == np.min(
        df_B_fit['Reduced Chi2'])][0]

    fit_B_params = {}
    for param in df_B_fit.columns:
        fit_B_params[param] = df_B_fit[param][idx_of_BF_B]



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
    for mol in mols:
        sample_model_in_uvplane(modelPath + '_bestFit-' + mol, mol)
        sample_model_in_uvplane(modelPath + '_bestFit-' + mol, mol, option='subtract')
        icr(modelPath + '_bestFit-' + mol, mol=mol)
        icr(modelPath + '_bestFit_resid-' + mol, mol=mol)
    print "Best-fit models created: " + modelPath + "_bestFit-[mol].im\n\n"

    # Calculate and present the final X2 values.
    finalX2s = chiSq(modelPath + '_bestFit', mol)
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
        s0 = '\nLOG FOR RUN ON' + today + ' FOR THE A FOUR LINE FIT'
        s1 = '\nBest Chi-Squared values [raw, reduced]:\n' + str(finalX2s)
        s2 = '\n\n\nParameter ranges queried:\n'
        s3 = '\nDisk A:\n'
        for i, ps in enumerate(diskAParams_fourline):
            s3 = s3 + param_names[i] + str(ps) + '\n'
        s4 = '\nDisk B:\n'
        for i, ps in enumerate(diskBParams_fourline):
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
