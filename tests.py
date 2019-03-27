"""
So try running a grid with disk B xmol fixed at something reasonable (in -7 to -4)
to see if anything changes. If it doesn't, then that means it might just be
optically thick (can check by looking at degeneracy map.)

"""


import mcmc
import pandas as pd
import numpy as np
import subprocess as sp
from utils import makeModel, sumDisks, chiSq
from tools import plot_fits, plot_spectrum, already_exists, remove, sample_model_in_uvplane, icr
from fitting import Model, Observation
from analysis import GridSearch_Run
from pathlib2 import Path
from constants import lines, obs_stuff, offsets, today
from run_params import make_diskA_params, make_diskB_params
from run_driver import make_fits, param_dict
from run_params import make_diskA_params

import matplotlib.pyplot as plt
from astropy.io import fits

Path.cwd()
modeling = '/Volumes/disks/jonas/modeling'
# %matplotlib inline

mol = 'co'

diskAParams = make_diskA_params(mol=mol, run_length='short')
diskBParams = make_diskB_params(mol=mol, run_length='short')

vsys, restfreq, freqs, obsv, chanstep, n_chans, chanmins, jnum = obs_stuff(mol)
pos_A, pos_B = offsets


# param_dict = {
#     'r_out_A':              400,             # AU
#     'r_out_B':              200,             # AU
#     'atms_temp_A':          300,
#     'atms_temp_B':          200,
#     'mol_abundance_A':      -10,
#     'mol_abundance_B':      -10,
#     'temp_struct_A':        -0.2,
#     'temp_struct_B':        -0.2,
#     'incl_A':               65.,
#     'incl_B':               45,
#     'pos_angle_A':          69.7,
#     'pos_angle_B':          136.,
#     'T_mids':              [15, 15],          # Kelvin
#     'r_ins':               1,                 # AU
#     'r_ins':               [1, 1],            # AU
#     'm_disk_A':            -1.10791,          # Disk Gas Masses (log10 solar masses)
#     'm_disk_B':            -1.552842,         # Disk Gas Masses (log10 solar masses)
#     'm_stars':             [3.5, 0.4],        # Solar masses (Disk A, B)
#     'surf_dens_str_A':     1.,                # Surface density power law index
#     'surf_dens_str_B':     1.,                # Surface density power law index
#     'v_turb':              0.081,             # Turbulence velocity
#     'vert_temp_str':       70.,               # Zq in Kevin's docs
#     'r_crit':              100.,              # Critical radius (AU)
#     'rot_hands':           [-1, -1],          # disk rotation direction
#     'distance':            389.143,           # parsec, errors of 3ish
#     'imres':               0.045,             # arcsec/pixel
#     'imwidth':             256,               # width of image (pixels)
#     'mol':                 mol,
#     'vsys':                vsys,              # km/s
#     'obsv':                obsv,              # km/s
#     'nchans':              n_chans,
#     'chanmins':            chanmins,
#     'restfreq':            restfreq,	   	  # GHz
#     'offsets':             [pos_A, pos_B],    # from center (")
#     'chanstep':            chanstep,
#     'jnum':                lines[mol]['jnum'],
#     'column_densities':    lines[mol]['col_dens'],
#     'T_freezeout':         lines[mol]['t_fo']
#     }

# Not sure where this came from but it looks like a fit result?
# testdict_fit = {'rot_hands' :  [-1, -1],
#             'r_crit' :  100.0,
#             'column_densities' :  [1.3e21/(1.59e21), 1e30/(1.59e21)],
#             'r_out_A' :  770.910089564,
#             'restfreq' :  356.734223,
#             'r_out_B' :  592.91782106,
#             'T_mids' :  [15, 15],
#             'chanstep' :  0.410341794388,
#             'mol' :  'hco',
#             'surf_dens_str_A' :  1.0,
#             'm_disk_B' :  -1.552842,
#             'surf_dens_str_B' :  1.0,
#             'm_stars' :  [3.5, 0.4],
#             'imwidth' :  256,
#             'r_ins' :  [1, 1],
#             'chanmins' :  chanmins,
#             'incl_B' :  12.1070550098,
#             'incl_A' :  67.5001509446,
#             'atms_temp_B' :  81.3680085166,
#             'T_freezeout' :  19,
#             'jnum' :  lines[mol]['jnum'],
#             'atms_temp_A' :  366.878245658,
#             'vsys' :  [10.0, 10.75],
#             'imres' :  0.045,
#             'm_disk_A' :  -1.10791,
#             'mol_abundance_B' :  -8.65404254293,
#             'offsets' :  [[0.0002, 0.082], [-1.006, -0.3]],
#             'mol_abundance_A' :  -7.84839263903,
#             'pos_angle_A' :  57.528424483,
#             'pos_angle_B' :  155.778624035,
#             'v_turb' :  0.081,
#             'vert_temp_str' :  70.0,
#             'temp_struct_B' :  -0.916682600684,
#             'nchans' :  [65, 61],
#             'temp_struct_A' :  1.40565384915,
#             'distance' :  389.143,
#             'obsv': obsv}


def get_param_dict(mol):
    diskAParams = make_diskA_params(mol=mol, run_length='short')
    diskBParams = make_diskB_params(mol=mol, run_length='short')

    vsys, restfreq, freqs, obsv, chanstep, n_chans, chanmins, jnum = obs_stuff(mol)
    pos_A, pos_B = offsets

    testdict = {'mol' :             mol,
                'vsys' :            vsys,
                'chanstep' :        chanstep,
                'chanmins' :        chanmins,
                'nchans' :          n_chans,
                'offsets' :         offsets,
                'obsv':             obsv,
                'r_out_A' :         diskAParams['r_out'],
                'r_out_B' :         diskBParams['r_out'],
                'm_disk_A' :        diskAParams['m_disk'],
                'm_disk_B' :        diskBParams['m_disk'],
                'incl_A' :          diskAParams['incl'],
                'incl_B' :          diskBParams['incl'],
                'atms_temp_A' :     diskAParams['t_atms'],
                'atms_temp_B' :     diskBParams['t_atms'],
                'mol_abundance_A' : diskAParams['x_mol'],
                'mol_abundance_B' : diskBParams['x_mol'],
                'pos_angle_A' :     diskAParams['PA'],
                'pos_angle_B' :     diskBParams['PA'],
                'temp_struct_A' :   diskAParams['t_qq'],
                'temp_struct_B' :   diskBParams['t_qq'],
                'vert_temp_str' :   diskAParams['zq'],
                'surf_dens_str_A' : diskAParams['rho_p'],
                'surf_dens_str_B' : diskBParams['rho_p'],
                'restfreq' :        lines[mol]['restfreq'],
                'column_densities': lines[mol]['col_dens'],
                'T_freezeout' :     lines[mol]['t_fo'],
                'jnum' :            lines[mol]['jnum'],
                'rot_hands' :   [-1, -1],
                'r_crit' :      100.0,
                'T_mids' :      [15, 15],
                'm_stars' :     [3.5, 0.4],
                'imwidth' :     256,
                'imres' :       0.045,
                'v_turb' :      0.081,
                'distance' :    389.143,
                'r_ins' :       [1, 1]
                }

    return testdict


# gs_testdict_a = {'v_turb': testdict['v_turb'],
#                  'zq': testdict['vert_temp_str'],
#                  'r_crit': testdict['r_crit'],
#                  'rho_p': testdict['surf_dens_str_A'],
#                  't_mid': testdict['T_mids'][0],
#                  'PA': testdict['pos_angle_A'],
#                  'incl': testdict['incl_A'],
#                  'pos_x': testdict['offsets'][0][0],
#                  'pos_y': testdict['offsets'][0][1],
#                  'v_sys': testdict['vsys'][0],
#                  't_atms': testdict['atms_temp_A'],
#                  't_qq': testdict['temp_struct_A'],
#                  'r_out': testdict['r_out_A'],
#                  'm_disk': testdict['m_disk_A'],
#                  'x_mol': testdict['mol_abundance_A']
#                  }

# gs_testdict_b = {'v_turb': testdict['v_turb'],
#                  'zq': testdict['vert_temp_str'],
#                  'r_crit': testdict['r_crit'],
#                  'rho_p': testdict['surf_dens_str_B'],
#                  't_mid': testdict['T_mids'][1],
#                  'PA': testdict['pos_angle_B'],
#                  'incl': testdict['incl_B'],
#                  'pos_x': testdict['offsets'][1][0],
#                  'pos_y': testdict['offsets'][1][1],
#                  'v_sys': testdict['vsys'][1],
#                  't_atms': testdict['atms_temp_B'],
#                  't_qq': testdict['temp_struct_B'],
#                  'r_out': testdict['r_out_B'],
#                  'm_disk': testdict['m_disk_B'],
#                  'x_mol': testdict['mol_abundance_B']
#                  }



def test_make_fits(mol, param_dict, save=True):

    base_fname = '{}-test'.format(mol)
    path='fits_by_hand/'
    fname, counter = base_fname, 1
    while already_exists(path + fname + '_image.pdf') is True:
        fname = '{}-{}'.format(base_fname, str(counter))
        counter += 1

    obs = Observation(mol)
    model = Model(obs, path, fname, testing=True)
    make_fits(model, param_dict, mol=mol)
    model.obs_sample()
    icr(path + fname, mol=mol)
    plot_spectrum(path + fname + '.fits', save=save)
    plot_fits(path + fname + '.fits', save=save)

    data_img = 'data/{}/images/{}-short{}_image.pdf'.format(mol, mol,
                                                            lines[mol]['baseline_cutoff'])
    data_spec = 'data/{}/images/{}-short{}_spectrum.pdf'.format(mol, mol,
                                                                lines[mol]['baseline_cutoff'])
    sp.call(['cp', data_img, path])
    sp.call(['cp', data_spec, path])

    remove([path + fname + ext for ext in ['.im', '.fits', '.vis', '.uvf']])


# diskAParams = make_diskA_params(mol='hco', run_length='short')
# diskBParams = make_diskB_params(mol='hco', run_length='short')

def test_makeModel(mol, diskAParams, diskBParams, save=False):

    fname = 'mm_test'
    base_path='./test_files/makeModel_test-{}/'.format(today)
    path, counter = base_path, 1
    while already_exists(path) is True:
        path = '{}-{}/'.format(base_path, str(counter))
        counter += 1
    sp.call(['mkdir', path])

    makeModel(diskAParams, path + fname + 'A', 0, mol)
    makeModel(diskBParams, path + fname + 'B', 1, mol)
    sumDisks(path + 'A', path + fname + 'B', path + fname + '_both', mol)
    plot_spectrum(path + fname + '_both.fits', save=True)
    sample_model_in_uvplane(path + fname + '_both', mol)






# test_make_fits('hco', testdict)
# test_make_fits('hcn')
# test_make_fits('co', save=True)
# test_make_fits('cs', save=True)
# test_makeModel('hco')

# mol = 'hco'
# obs = Observation(mol)
# model = Model(obs, 'jan22-3', 'jan22-3')

# Name     = (mol,   param,   DI, vals_to_try)
# co_tatms_A = ('co', 't_atms', 0, range(50, 400, 25))
# co_tatms_B = ('co', 't_atms', 1, range(50, 400, 25))
# co_xmol_A = ('co', 'xmol', 0, range(-11, -3, 1))
# co_xmol_B = ('co', 'xmol', 1, range(-11, -3, 1))

def gs_test_param_dependence(vector):
    mol, param, DI, param_vals = vector

    path = 'param_dependence_testing/{}-{}-disk{}/'.format(mol, param, DI)
    fname = 'disk_'

    remove(path)
    sp.call(['mkdir', path])
    print('Made new directory at ' + path)

    # Generate params to be used.
    diskAParams = make_diskA_params(mol, run_length='short')
    diskBParams = make_diskB_params(mol, run_length='short')
    # make_diskX_params() generates lists, so just extract the values.
    for k in list(diskAParams.keys()):
        diskAParams[k] = diskAParams[k][0]
        diskBParams[k] = diskBParams[k][0]

    chisq_list = []
    for p in param_vals:
        print('\n\n\n')
        print('Evaluating {} = {} for DI = {}'.format(param, p, DI))

        # Edit the appropriate element.
        if DI == 0:
            diskAParams[param] = p
            print(diskAParams[param])
        else:
            diskBParams[param] = p
            print(diskBParams[param])
        # Make a model.
        makeModel(diskBParams, path + fname + 'B', 1, mol)
        makeModel(diskAParams, path + fname + 'A', 0, mol)
        sumDisks(path + fname + 'A', path + fname + 'B', path + fname + 'both', mol)
        sample_model_in_uvplane(path + fname + 'both', mol)
        chisq_val = chiSq(path + fname + 'both', mol)
        chisq_list.append(chisq_val)

        print("Chi2 Value:", chisq_val)

    plt.plot(param_vals, chisq_list, 'or')
    plt.plot(param_vals, chisq_list, '-r')
    plt.xlabel(param, weight='bold')
    plt.ylabel('Chi2 Value', weight='bold')
    plt.savefig(path + 'chisq_plot.pdf')
    print("Saved figure to {}".format(path + 'chisq_plot.pdf'))

    df = pd.DataFrame({'chisqs': chisq_list, 'param_vals': param_vals})
    df.to_csv('chisq_df.csv')

    return df


test_vector = ('co', 'atms_temp_A', [100, 500])


co_tatms_A_model = ('co', 'atms_temp_A', list(range(50, 400, 40)))
co_tatms_B_model = ('co', 'atms_temp_B', list(range(50, 400, 40)))

co_mdisk_A_model = ('co', 'm_disk_A', list(np.arange(-1.25, -0.95, 0.04)))
co_mdisk_B_model = ('co', 'm_disk_B', list(np.arange(-2.1, -1.7, 0.05)))

co_tatms_A_model_reallylow = ('co', 'atms_temp_A', list(range(0, 50, 10)))
co_tatms_B_model_low = ('co', 'atms_temp_B', list(range(50, 150, 10)))

co_mdisk_A_model_mid = ('co', 'm_disk_A', list(np.arange(-2.2, -1.2, 0.1)))
co_mdisk_B_model_low = ('co', 'm_disk_B', list(np.arange(-7., -3.5, 0.5)))

def test_param_dependence_model(vector):
    mol, param, param_vals = vector

    path = 'param_dependence_testing_model/{}-{}/'.format(mol, param)
    fname = 'disk_'

    remove(path)
    sp.call(['mkdir', path])
    print('Made new directory at ' + path)

    # Generate params to be used.
    param_dict = get_param_dict(mol)

    chisq_list = []
    for p in param_vals:
        print('\n\n\n')
        print('Evaluating {} = {}'.format(param, p))

        # Edit the appropriate element.
        param_dict[param] = p

        # Make a model.
        obs = Observation(mol)
        m = Model(obs, path, fname + 'both', testing=True)
        make_fits(m, param_dict, mol)
        m.obs_sample()
        # m.chiSq takes mol as an argument to tell if it should calculate for central
        # chans (only does so if mol == 'co'). Don't want that so we can compare directly to utils.chiSq
        chisq_val = m.chiSq('co')[0]
        m.delete()

        chisq_list.append(chisq_val)
        print("Chi2 Value:", chisq_val)

    delta_chisqs = np.array(chisq_list) - min(chisq_list)
    plt.plot(param_vals, delta_chisqs, 'or')
    plt.plot(param_vals, delta_chisqs, '-r')
    plt.xlabel(param, weight='bold')
    # plt.ylabel(r"$\chi^2$ - $\chi_{\text{min}}^2$", weight='bold')
    plt.ylabel('Chi2 - Min(Chi2)', weight='bold')
    plt.title('Chi2 Vals for {} in {}'.format(param, mol.upper()), weight='bold')
    plt.savefig(path + 'chisq_plot.pdf')
    print("Saved figure to {}".format(path + 'chisq_plot.pdf'))
    plt.clf()

    df = pd.DataFrame({'chisqs': chisq_list, 'param_vals': param_vals})
    df.to_csv(path + 'chisq_df.csv')

    return df














# The End
