"""
So try running a grid with disk B xmol fixed at something reasonable (in -7 to -4)
to see if anything changes. If it doesn't, then that means it might just be
optically thick (can check by looking at degeneracy map.)

"""


import mcmc
import numpy as np
import subprocess as sp
from utils import makeModel, sumDisks
from tools import plot_fits, plot_spectrum
from fitting import Model, Observation
from analysis import GridSearch_Run
from pathlib2 import Path
from constants import lines, obs_stuff, offsets
from run_params import make_diskA_params, make_diskB_params
from run_driver import make_fits, param_dict
from run_params import make_diskA_params

from astropy.io import fits

Path.cwd()
modeling = '/Volumes/disks/jonas/modeling'
# %matplotlib inline

mol = 'hco'

diskAParams = make_diskA_params(mol=mol, run_length='short')

vsys, restfreq, freqs, obsv, chanstep, n_chans, chanmins, jnum = obs_stuff(mol)
pos_A, pos_B = offsets
chanmins

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
    'chanstep':            (1) * np.abs(obsv[1] - obsv[0]),
    'jnum':                lines[mol]['jnum'],
    'column_densities':    lines[mol]['col_dens'],
    'T_freezeout':         lines[mol]['t_fo']
    }


testdict = {'rot_hands' :  [-1, -1],
            'r_crit' :  100.0,
            'column_densities' :  [0.8176100628930818, 628930817.610063],
            'r_out_A' :  770.910089564,
            'restfreq' :  356.734223,
            'r_out_B' :  592.91782106,
            'T_mids' :  [15, 15],
            'chanstep' :  0.410341794388,
            'mol' :  'hco',
            'surf_dens_str_A' :  1.0,
            'm_disk_B' :  -1.552842,
            'surf_dens_str_B' :  1.0,
            'm_stars' :  [3.5, 0.4],
            'imwidth' :  256,
            'r_ins' :  [1, 1],
            'chanmins' :  [23.13093742042156, 23.060253831645213],
            'incl_B' :  12.1070550098,
            'incl_A' :  67.5001509446,
            'atms_temp_B' :  81.3680085166,
            'T_freezeout' :  19,
            'jnum' :  3,
            'atms_temp_A' :  366.878245658,
            'vsys' :  [10.0, 10.75],
            'imres' :  0.045,
            'm_disk_A' :  -1.10791,
            'mol_abundance_B' :  -8.65404254293,
            'offsets' :  [[0.0002, 0.082], [-1.006, -0.3]],
            'mol_abundance_A' :  -7.84839263903,
            'pos_angle_A' :  57.528424483,
            'pos_angle_B' :  155.778624035,
            'v_turb' :  0.081,
            'vert_temp_str' :  70.0,
            'temp_struct_B' :  -0.916682600684,
            'nchans' :  [65, 61],
            'temp_struct_A' :  1.40565384915,
            'distance' :  389.143,
            'obsv': obsv}

gs_testdict_a = {'v_turb': testdict['v_turb'],
                 'zq': testdict['vert_temp_str'],
                 'r_crit': testdict['r_crit'],
                 'rho_p': testdict['surf_dens_str_A'],
                 't_mid': testdict['T_mids'][0],
                 'PA': testdict['pos_angle_A'],
                 'incl': testdict['incl_A'],
                 'pos_x': testdict['offsets'][0][0],
                 'pos_y': testdict['offsets'][0][1],
                 'v_sys': testdict['vsys'][0],
                 't_atms': testdict['atms_temp_A'],
                 't_qq': testdict['temp_struct_A'],
                 'r_out': testdict['r_out_A'],
                 'm_disk': testdict['m_disk_A'],
                 'x_mol': testdict['mol_abundance_A']
                 }

for p in gs_testdict_a.keys():
    print p, gs_testdict_a[p]





gs_testdict_b = {'v_turb': testdict['v_turb'],
                 'zq': testdict['vert_temp_str'],
                 'r_crit': testdict['r_crit'],
                 'rho_p': testdict['surf_dens_str_B'],
                 't_mid': testdict['T_mids'][1],
                 'PA': testdict['pos_angle_B'],
                 'incl': testdict['incl_B'],
                 'pos_x': testdict['offsets'][1][0],
                 'pos_y': testdict['offsets'][1][1],
                 'v_sys': testdict['vsys'][1],
                 't_atms': testdict['atms_temp_B'],
                 't_qq': testdict['temp_struct_B'],
                 'r_out': testdict['r_out_B'],
                 'm_disk': testdict['m_disk_B'],
                 'x_mol': testdict['mol_abundance_B']
                 }



def test_make_fits(mol, param_dict, save=True):
    obs = Observation(mol)
    sp.call(['mkdir', 'mcmc_runs/jan23_test_{}'.format(mol)])
    sp.call(['mkdir', 'mcmc_runs/jan23_test_{}/model_files/'.format(mol)])
    print "Made directories"
    model = Model(obs, 'jan23_test_' + mol, 'test1')
    make_fits(model, param_dict, mol=mol)
    model.obs_sample()
    print "Finished making fits files; plotting now"
    plot_spectrum('/scratch/jonas/mcmc_runs/jan23_test_{}/model_files/test1.fits'.format(mol), save=save)
    plot_fits('/scratch/jonas/mcmc_runs/jan23_test_{}/model_files/test1.fits'.format(mol), save=save)



diskAParams = make_diskA_params(mol='hco', run_length='short')
diskBParams = make_diskB_params(mol='hco', run_length='short')

def test_makeModel(mol, diskAParams, diskBParams, save=False):

    path='./test_files/makeModel'
    makeModel(diskAParams, path + '_testA', 0, mol)
    makeModel(diskBParams, path + '_testB', 1, mol)
    sumDisks(path + '_testA', path + '_testB', path + '_test_both', mol)
    plot_spectrum(path + '_test_both.fits', save=True)
    sample_model_in_uvplane(path + '_test_both', mol)





# test_make_fits('hco', testdict)
# test_make_fits('hcn')
# test_make_fits('co', save=True)
# test_make_fits('cs', save=True)
# test_makeModel('hco')

mol = 'hco'
obs = Observation(mol)
model = Model(obs, 'jan22-3', 'jan22-3')









# The End
