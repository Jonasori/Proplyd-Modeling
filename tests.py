import mcmc
import numpy as np
from analysis import GridSearch_Run
from pathlib2 import Path
from run_driver import make_fits, param_dict
from run_params import make_diskA_params
from fitting import Model, Observation
from tools import plot_fits

from astropy.io import fits




diskAParams = make_diskA_params(mol=mol, run_length='long')

Path.cwd()
modeling = '/Volumes/disks/jonas/modeling'
%matplotlib inline

param_dict = {
    'r_out_A':              400,        # AU
    'r_out_B':              200,        # AU
    'atms_temp_A':          300,
    'atms_temp_B':          200,
    'mol_abundance_A':      -6.,
    'mol_abundance_B':      -6.,
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
    'imwidth':             256,               # width of image (pixels)
    'mol':                 mol                # Emission line
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


test_make_fits('hco')
test_make_fits('hcn')
test_make_fits('co', save=True)
test_make_fits('cs', save=True)


plot_fits('/scratch/jonas/mcmc_runs/jan16_test_cs/model_files/test1.fits')


run = mcmc.MCMCrun('mcmc_runs/jan21/', 'jan21')
run.main






# The End
