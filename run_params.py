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

# Big run values
"""
    TatmsA       = np.arange(10, 500, 50)
    TqqA         = np.array([-0.5, 0, 0.5])
    R_outA       = np.arange(50, 500, 50)
    PAA          = np.array([69.7])
    InclA        = np.array([65])
    Pos_XA       = np.array([pos_Ax])
    Pos_YA       = np.array([pos_Ay])
    VsysA        = np.array([vsysA])
    # Note that both these are log-scaled and in m_disk is in units of solar masses
    if mol == 'co':
        MdiskA   = np.arange(-1.4, -0.8, 0.1)
        XmolA    = -1 * np.array([4.])
    else:
        MdiskA   = np.array([-1.10791])
        XmolA    = -1 * np.arange(6., 13., 1.)


    # Parameters for Disk B
    TatmsB       = np.arange(10, 500, 50)
    TqqB         = -1 * np.array([-0.5, 0, 0.5])
    R_outB       = np.arange(50, 400, 50)
    PAB          = np.array([135])
    InclB        = np.array([30])
    Pos_XB       = np.array([pos_Bx])
    Pos_YB       = np.array([pos_By])
    VsysB        = np.array([10.70])
    # Note that both these are log-scaled and in m_disk is in units of solar masses
    if mol == 'co':
        MdiskB   = np.arange(-1.8, -1.3, 0.1)
        XmolB    = -1 * np.array([4.])
    else:
        MdiskB   = np.array([-1.552842])
        XmolB    = -1 * np.arange(6., 13., 1.)
    """



def make_diskB_params(mol='hco', run_length='mid'):
    # Params that are always constant
    v_turb       = np.array([0.081])
    zq           = np.array([70.])
    r_crit       = np.array([100.])
    rho_p        = np.array([1.])
    t_mid        = np.array([15.])

    PA          = np.array([135])
    incl        = np.array([30])
    pos_x       = np.array([offsets[1][0]])
    pos_y       = np.array([offsets[1][1]])
    v_sys        = vsys[1]          # np.array([10.70])
    m_disk   = np.array([-1.552842])

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
            m_disk = np.array([-1.552842])
        elif run_length == 'long':
            m_disk = np.array([-1.552842])

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
              't_atms': t_qq,
              'r_out': r_out,
              'm_disk': m_disk,
              'x_mol': x_mol}
    return params


# diskAParams = make_diskA_params(mol=mol, run_length='mid')
diskBParams = make_diskB_params(mol=mol, run_length='mid')
