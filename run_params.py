"""Some sets of parameters to be grid searching over."""

import numpy as np
from constants import offsets, vsys, mol
"""
# Offset vals from original fitting
pos_Ax = -0.0298
pos_Ay = 0.072
pos_Bx = -1.0456
pos_By = -0.1879
"""


# ONE LINE PARAMS

def make_diskA_params(mol, run_length='long'):
    # Params that are constant but could be fit
    v_turb       = np.array([0.081])
    zq           = np.array([29.])      # Was formerly 70, but Sam fixed at 29 for all lines
    r_crit       = np.array([100.])
    rho_p        = np.array([1.])
    t_mid        = np.array([19.])
    PA           = np.array([69.7])
    incl         = np.array([65])
    pos_x        = np.array([offsets[0][0]])
    pos_y        = np.array([offsets[0][1]])
    v_sys        = [vsys[0]]          # np.array([10.70])
    m_disk       = np.array([-1.10791])
    t_qq         = np.array([-0.5])

    # Params that are fit
    if run_length == 'short':
        t_atms      = np.array([100])
        r_out       = np.array([500])
        x_mol       = np.array([-4.])
    elif run_length == 'mid':
        t_atms      = np.array([10, 200])
        r_out       = np.array([100, 300, 600])
        x_mol       = -1. * np.array([6, 9])

    elif run_length == 'long':
        t_atms      = np.arange(25, 400, 25)
        r_out       = np.arange(200, 800, 50)
        x_mol       = -1. * np.arange(2, 10)

    else:
        return "Please choose 'short', 'mid', or 'long'"

    # If we're looking at CO, then fix X_mol and fit for M_disk
    if mol == 'co':
        x_mol = np.array([-4.])
        if run_length == 'short':
            m_disk = np.array([-1.10791])
        elif run_length == 'mid':
            m_disk = np.array([-1.07, -1.108, -1.109])
        elif run_length == 'long':
            m_disk = np.array([-1.104, -1.106, -1.108, -1.11, -1.112])

    params = np.array([v_turb, zq, r_crit, rho_p, t_mid, PA, incl,
                       pos_x, pos_y, v_sys, t_atms, t_qq, r_out,
                       m_disk, x_mol])
    params = {'v_turb': v_turb,
              'zq': zq,
              'r_crit': r_crit,
              'rho_p': rho_p,
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
              'x_mol': x_mol}
    return params


def make_diskB_params(mol, run_length='long'):
    # Params that are constant but could be fit
    v_turb       = np.array([0.081])
    zq           = np.array([29.])
    r_crit       = np.array([100.])
    rho_p        = np.array([1.])
    t_mid        = np.array([15.])
    PA           = np.array([135])
    incl         = np.array([30])
    pos_x        = np.array([offsets[1][0]])
    pos_y        = np.array([offsets[1][1]])
    v_sys        = [vsys[1]]          # np.array([10.70])
    m_disk       = np.array([-1.552842])
    t_qq         = np.array([-0.5])

    # Params that are fit
    if run_length == 'short':
        t_atms      = np.array([100])
        r_out       = np.array([400])
        x_mol       = np.array([-4.])
    elif run_length == 'mid':
        t_atms      = np.arange(50, 500, 100)
        r_out       = np.arange(100, 600, 100)
        x_mol       = np.array([-4.])
    elif run_length == 'long':
        t_atms      = np.arange(10, 300, 25)
        r_out       = np.arange(150, 550, 25)
        x_mol       = -1. * np.arange(2, 10)

    else:
        return "Please choose 'short', 'mid', or 'long'"

    # If we're looking at CO, then fix X_mol and fit for M_disk
    if mol == 'co':
        x_mol = np.array([-4.])
        if run_length == 'short':
            m_disk = np.array([-1.552842])
        elif run_length == 'mid':
            m_disk = np.array([-1.552, -1.553, -1.554])
        elif run_length == 'long':
            m_disk = np.array([-1.551, -1.552, -1.553, -1.554, -1.555])

    params = np.array([v_turb, zq, r_crit, rho_p, t_mid, PA, incl,
                       pos_x, pos_y, v_sys, t_atms, t_qq, r_out,
                       m_disk, x_mol])
    params = {'v_turb': v_turb,
              'zq': zq,
              'r_crit': r_crit,
              'rho_p': rho_p,
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
              'x_mol': x_mol}
    return params


# diskAParams = make_diskA_params(mol=mol, run_length='long')
# diskBParams = make_diskB_params(mol=mol, run_length='long')




# The End
