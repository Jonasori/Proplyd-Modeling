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

def make_diskA_params(mol='hco', run_length='mid'):
    # Params that are constant but could be fit
    v_turb       = np.array([0.081])
    zq           = np.array([70.])
    r_crit       = np.array([100.])
    rho_p        = np.array([1.])
    t_mid        = np.array([15.])
    PA           = np.array([69.7])
    incl         = np.array([65])
    pos_x        = np.array([offsets[0][0]])
    pos_y        = np.array([offsets[0][1]])
    v_sys        = vsys[0]          # np.array([10.70])
    m_disk       = np.array([-1.10791])

    # Params that are fit
    if run_length == 'short':
        t_atms      = np.array([100])
        t_qq        = -1 * np.array([0])
        r_out       = np.array([150])
        x_mol       = -1 * np.array([4.])
    elif run_length == 'mid':
        t_atms      = np.array([10, 200])
        t_qq        = -1 * np.array([-0.5, 0, 0.5])
        r_out       = np.array([10, 150, 400])
        x_mol       = -1 * np.array([6., 9.])
    elif run_length == 'long':
        t_atms      = np.arange(10, 500, 100)
        t_qq        = -1 * np.array([-0.5, 0, 0.5])
        r_out       = np.arange(50, 500, 50)
        x_mol       = -1 * np.array([4.])

    else:
        return "Please choose 'short', 'mid', or 'long'"

    # If we're looking at CO, then fix X_mol and fit for M_disk
    if mol == 'co':
        x_mol = -1 * np.array([4.])
        if run_length == 'short':
            m_disk = np.array([-1.10791])
        elif run_length == 'mid':
            m_disk = np.array([-1.07, -1.108, -1.109])
        elif run_length == 'long':
            m_disk = np.array([-1.06, -1.07, -1.108, -1.109, -1.11])

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


def make_diskB_params(mol='hco', run_length='mid'):
    # Params that are constant but could be fit
    v_turb       = np.array([0.081])
    zq           = np.array([70.])
    r_crit       = np.array([100.])
    rho_p        = np.array([1.])
    t_mid        = np.array([15.])
    PA           = np.array([135])
    incl         = np.array([30])
    pos_x        = np.array([offsets[1][0]])
    pos_y        = np.array([offsets[1][1]])
    v_sys        = vsys[1]          # np.array([10.70])
    m_disk       = np.array([-1.552842])

    # Params that are fit
    if run_length == 'short':
        t_atms      = np.array([100])
        t_qq        = -1 * np.array([0])
        r_out       = np.array([150])
        x_mol       = -1 * np.array([4.])
    elif run_length == 'mid':
        t_atms      = np.arange(10, 500, 100)
        t_qq        = -1 * np.array([-0.5, 0, 0.5])
        r_out       = np.arange(50, 400, 100)
        x_mol       = -1 * np.array([4.])
    elif run_length == 'long':
        t_atms      = np.arange(10, 500, 50)
        t_qq        = -1 * np.array([-0.5, 0, 0.5])
        r_out       = np.arange(50, 400, 50)
        x_mol       = -1 * np.array([4.])

    else:
        return "Please choose 'short', 'mid', or 'long'"

    # If we're looking at CO, then fix X_mol and fit for M_disk
    if mol == 'co':
        x_mol = -1 * np.array([4.])
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


diskAParams = make_diskA_params(mol=mol, run_length='mid')
diskBParams = make_diskB_params(mol=mol, run_length='mid')






# FOUR LINE PARAMS

def make_diskA_params_fourline(mol='hco'):
    # Params that are constant but could be fit
    v_turb       = np.array([0.081])
    r_crit       = np.array([100.])
    rho_p        = np.array([1.])
    t_mid        = np.array([15.])
    PA           = np.array([135])
    incl         = np.array([30])
    pos_x        = np.array([offsets[1][0]])
    pos_y        = np.array([offsets[1][1]])
    v_sys        = vsys[1]          # np.array([10.70])
    m_disk       = np.array([-1.552842])

    # Params that are fit
    t_qq        = -1 * np.array([-0.5, 0, 0.5])

    # Line-specific vals
    zq_co = np.array([70.])
    zq_cs = np.array([70.])
    zq_hco = np.array([70.])
    zq_hcn = np.array([70.])

    t_atms_co = np.array([10, 100, 300])
    t_atms_cs = np.array([10, 100, 300])
    t_atms_hco = np.array([10, 100, 300])
    t_atms_hcn = np.array([10, 100, 300])

    r_out_co = np.array([10, 200, 500])
    r_out_cs = np.array([10, 200, 500])
    r_out_hco = np.array([10, 200, 500])
    r_out_hcn = np.array([10, 200, 500])

    x_mol_co = -1 * np.array([4.])
    x_mol_cs = -1 * np.array([5., 9.])
    x_mol_hco = -1 * np.array([5., 9.])
    x_mol_hcn = -1 * np.array([5., 9.])

    m_disk_co = np.array([-1.552, -1.553, -1.554])
    m_disk_cs = np.array([-1.553])
    m_disk_hco = np.array([-1.553])
    m_disk_hcn = np.array([-1.553])


    params = {'v_turb': v_turb,
              'r_crit': r_crit,
              'rho_p': rho_p,
              't_mid': t_mid,
              'PA': PA,
              'incl': incl,
              'pos_x': pos_x,
              'pos_y': pos_y,
              'v_sys': v_sys,
              't_qq': t_qq,

              'r_out_co': r_out_co,
              'r_out_cs': r_out_cs,
              'r_out_hco': r_out_hco,
              'r_out_hcn': r_out_hcn,

              't_atms_co': t_atms_co,
              't_atms_cs': t_atms_cs,
              't_atms_hco': t_atms_hco,
              't_atms_hcn': t_atms_hcn,

              'x_mol_co': x_mol_co,
              'x_mol_cs': x_mol_cs,
              'x_mol_hco': x_mol_hco,
              'x_mol_hcn': x_mol_hcn,

              'zq_co': zq_co,
              'zq_cs': zq_cs,
              'zq_hco': zq_hco,
              'zq_hcn': zq_hcn,

              'm_disk_co': m_disk_co,
              'm_disk_cs': m_disk_cs,
              'm_disk_hco': m_disk_hco,
              'm_disk_hcn': m_disk_hcn,
              }
    return params


def make_diskB_params_fourline(mol='hco'):
    # Params that are constant but could be fit
    v_turb       = np.array([0.081])
    r_crit       = np.array([100.])
    rho_p        = np.array([1.])
    t_mid        = np.array([15.])
    PA           = np.array([135])
    incl         = np.array([30])
    pos_x        = np.array([offsets[1][0]])
    pos_y        = np.array([offsets[1][1]])
    v_sys        = vsys[1]          # np.array([10.70])
    m_disk       = np.array([-1.552842])

    # Params that are fit
    t_qq = -1 * np.array([-0.5, 0, 0.5])
    zq = np.array([70.])

    # Line-specific vals
    zq_co = np.array([70.])
    zq_cs = np.array([70.])
    zq_hco = np.array([70.])
    zq_hcn = np.array([70.])

    t_atms_co = np.array([10, 100, 300])
    t_atms_cs = np.array([10, 100, 300])
    t_atms_hco = np.array([10, 100, 300])
    t_atms_hcn = np.array([10, 100, 300])

    r_out_co = np.array([10, 200, 500])
    r_out_cs = np.array([10, 200, 500])
    r_out_hco = np.array([10, 200, 500])
    r_out_hcn = np.array([10, 200, 500])

    x_mol_co = -1 * np.array([4.])
    x_mol_cs = -1 * np.array([5., 9.])
    x_mol_hco = -1 * np.array([5., 9.])
    x_mol_hcn = -1 * np.array([5., 9.])

    m_disk_co = np.array([-1.552, -1.553, -1.554])
    m_disk_cs = np.array([-1.553])
    m_disk_hco = np.array([-1.553])
    m_disk_hcn = np.array([-1.553])


    params = {'v_turb': v_turb,
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

              'r_out_co': r_out_co,
              'r_out_cs': r_out_cs,
              'r_out_hco': r_out_hco,
              'r_out_hcn': r_out_hcn,

              't_atms_co': t_atms_co,
              't_atms_cs': t_atms_cs,
              't_atms_hco': t_atms_hco,
              't_atms_hcn': t_atms_hcn,

              'x_mol_co': x_mol_co,
              'x_mol_cs': x_mol_cs,
              'x_mol_hco': x_mol_hco,
              'x_mol_hcn': x_mol_hcn,

              # 'zq_co': zq_co,
              # 'zq_cs': zq_cs,
              # 'zq_hco': zq_hco,
              # 'zq_hcn': zq_hcn,

              'm_disk_co': m_disk_co,
              'm_disk_cs': m_disk_cs,
              'm_disk_hco': m_disk_hco,
              'm_disk_hcn': m_disk_hcn,
              }
    return params


diskAParams_fourline = make_diskA_params_fourline(mol=mol)
diskBParams_fourline = make_diskB_params_fourline(mol=mol)
