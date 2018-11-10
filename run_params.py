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

# Fit offset vals:
pos_Ax = offsets[0][0]
pos_Ay = offsets[0][1]

pos_Bx = offsets[1][0]
pos_By = offsets[1][1]

vsysA, vsysB = vsys[0], vsys[1]


# Big run values
#"""
TatmsA       = np.arange(10, 500, 50)
TqqA         = np.array([-0.5, 0, 0.5])
R_outA       = np.arange(50, 500, 50)
# PA and InclA are from Williams et al
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
#"""


# Short run values
"""
TatmsA       = np.array([20, 200])
TqqA         = np.array([0])
R_outA       = np.array([100, 400])
# PA and InclA are from Williams et al
PAA          = np.array([69.7])
InclA        = np.array([65])
Pos_XA       = np.array([pos_Ax])
Pos_YA       = np.array([pos_Ay])
VsysA        = np.array([vsysA])
# Note that both these are log-scaled and in m_disk is in units of solar masses
if mol == 'co':
    MdiskA   = np.array([-1.10791, -0.9])
    XmolA    = -1 * np.array([4.])
else:
    MdiskA   = np.array([-1.10791])
    XmolA    = -1 * np.array([4., 6.])


# Parameters for Disk B
TatmsB       = np.array([20, 200])
TqqB         = -1 * np.array([0])
R_outB       = np.array([50, 300])
PAB          = np.array([135])
InclB        = np.array([30])
Pos_XB       = np.array([pos_Bx])
Pos_YB       = np.array([pos_By])
VsysB        = np.array([10.70])
# Note that both these are log-scaled and in m_disk is in units of solar masses
if mol == 'co':
    MdiskB   = np.array([-1.552842, -1.])
    XmolB    = -1 * np.array([4.])
else:
    MdiskB   = np.array([-1.552842])
    XmolB    = -1 * np.array([4., 6.])
"""


# One-step run values
"""
TatmsA       = np.array([100])
TqqA         = np.array([0])
R_outA       = np.array([300])
# PA and InclA are from Williams et al
PAA          = np.array([69.7])
InclA        = np.array([65])
Pos_XA       = np.array([pos_Ax])
Pos_YA       = np.array([pos_Ay])
VsysA        = np.array([vsysA])
# Note that both these are log-scaled and in m_disk is in units of solar masses
if mol == 'co':
    MdiskA   = np.array([-1.10791])
    XmolA    = -1 * np.array([4.])
else:
    MdiskA   = np.array([-1.10791])
    XmolA    = -1 * np.array([4.])


# Parameters for Disk B
TatmsB       = np.array([100])
TqqB         = -1 * np.array([0])
R_outB       = np.array([200])
PAB          = np.array([135])
InclB        = np.array([30])
Pos_XB       = np.array([pos_Bx])
Pos_YB       = np.array([pos_By])
VsysB        = np.array([10.70])
# Note that both these are log-scaled and in m_disk is in units of solar masses
if mol == 'co':
    MdiskB   = np.array([-1.552842])
    XmolB    = -1 * np.array([4.])
else:
    MdiskB   = np.array([-1.552842])
    XmolB    = -1 * np.array([4.])
"""










diskAParams = np.array([TatmsA, TqqA, XmolA, R_outA, PAA, InclA,
                        Pos_XA, Pos_YA, VsysA, MdiskA])
diskBParams = np.array([TatmsB, TqqB, XmolB, R_outB, PAB, InclB,
                        Pos_XB, Pos_YB, VsysB, MdiskB])
