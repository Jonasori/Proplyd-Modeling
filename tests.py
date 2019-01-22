"""
So try running a grid with disk B xmol fixed at something reasonable (in -7 to -4)
to see if anything changes. If it doesn't, then that means it might just be
optically thick (can check by looking at degeneracy map.)

"""






import mcmc
import numpy as np
from analysis import GridSearch_Run
from pathlib2 import Path
from utils import makeModel, sumDisks
from constants import lines, obs_stuff, offsets
from run_params import make_diskA_params, make_diskB_params
from run_driver import make_fits, param_dict
from run_params import make_diskA_params
from fitting import Model, Observation
from tools import plot_fits
from astropy.io import fits

Path.cwd()
modeling = '/Volumes/disks/jonas/modeling'
%matplotlib inline

mol = 'hco'
lines['hco']

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

def test_make_fits(mol, save=False):
    obs = Observation(mol)
    sp.call(['mkdir', 'mcmc_runs/jan16_test_{}'.format(mol)])
    sp.call(['mkdir', 'mcmc_runs/jan16_test_{}/model_files/'.format(mol)])
    print "Made directories"
    model = Model(obs, 'jan16_test_' + mol, 'test1')
    make_fits(model, param_dict, mol=mol)
    print "Finished making fits files; plotting now"
    plot_fits('/scratch/jonas/mcmc_runs/jan16_test_{}/model_files/test1.fits'.format(mol), save=save)

def test_makeModel(mol, save=False):
    diskAParams = make_diskA_params(mol='hco', run_length='short')
    diskBParams = make_diskB_params(mol='hco', run_length='short')

    makeModel(diskAParams, './test_files/makeModel_testA.fits', 0, mol)
    makeModel(diskBParams, './test_files/makeModel_testB.fits', 1, mol)
    sumDisks('./test_files/makeModel_testA', './test_files/makeModel_testB', './test_files/makeModel_test_both', mol)

# test_make_fits('hco')
# test_make_fits('hcn')
# test_make_fits('co', save=True)
# test_make_fits('cs', save=True)


test_makeModel('hco')

mol = 'hco'
obs = Observation(mol)
model = Model(obs, 'jan22-3', 'jan22-3')
    for p in param_dict.keys():
        if p != 'obsv':
            print p, ': ', param_dict[p]







# The End
