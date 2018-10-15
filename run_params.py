"""
Some sets of parameters to be grid searching over.

Only fit atmsT, Xmol, RA
"""

import numpy as np
from constants import offsets, vsys
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
TatmsA    = np.arange(150, 500, 50)
# TqqA    = -1 * np.array([0.5])
TqqA      = np.array([-0.5])
XmolA     =  -1 * np.arange(7.7, 9.7, 0.3)
R_outA    = np.arange(200, 500, 50)
# PA and InclA are from Williams et al
PAA       = np.array([69.7])
InclA     = np.arange(55, 75, 5)
Pos_XA    = np.array([pos_Ax])
Pos_YA    = np.array([pos_Ay])
VsysA     = np.array([vsysA])

# Parameters for Disk B
TatmsB    = np.array([400])
TqqB      = -1 * np.array([-0.5])
XmolB     = -1 * np.array([9.])
R_outB    = np.array([150])
PAB       = np.array([135])
InclB     = np.array([30])
Pos_XB    = np.array([pos_Bx])
Pos_YB    = np.array([pos_By])
VsysB     = np.array([10.70])

#"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Medium-length test run values
"""
# Parameters for Disk B
TatmsA    = np.arange(75, 500, 75)
# TqqA    = -1 * np.array([0.5])
TqqA      = np.array([-0.5])
XmolA     =  -1 * np.arange(6.5, 10., 0.5)
R_outA    = np.arange(100, 800, 100)
# PA and InclA are from Williams et al
PAA       = np.array([69.7])
InclA     = np.array([70])
Pos_XA    = np.array([pos_Ax])
Pos_YA    = np.array([pos_Ay])
VsysA     = np.array([vsysA])
#Pos_XA    = np.arange(pos_Ax - 0.05, pos_Ax + 0.15, 0.05)
#Pos_YA    = np.arange(pos_Ay - 0.05, pos_Ay + 0.15, 0.05)
#VsysA     = np.arange(vsysA - 0.11, vsysA + 0.1, 0.03)

# Parameters for Disk B
TatmsB    = np.arange(75, 500, 75)
TqqB      = np.array([-0.5])
XmolB     = -1 * np.arange(6.5, 10., 0.5)
R_outB    = np.arange(50, 400, 100)
PAB       = np.array([135])
InclB     = np.array([45])
Pos_XB    = np.array([pos_Bx])
Pos_YB    = np.array([pos_By])
VsysB     = np.array([vsysB])
#Pos_XB    = np.arange(pos_Bx - 0.1, pos_Bx + 0.1, 0.05)
#Pos_YB    = np.arange(pos_By - 0.1, pos_By + 0.2, 0.05)
#VsysB     = np.arange(vsysB - 0.1, vsysB + 0.2, 0.05)
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Short test run values

# Parameters for Disk A
"""
TatmsA    = np.array([300])
TqqA      = -1 * np.array([-0.5])
XmolA     = -1 * np.array([6.])
R_outA    = np.array([300])
# PA and InclA are from Williams et al
PAA       = np.array([69.7])
InclA     = np.array([65])
Pos_XA    = np.array([pos_Ax])
Pos_YA    = np.array([pos_Ay])
VsysA     = np.array([vsysA])


# Parameters for Disk B
TatmsB    = np.array([200])
TqqB      = -1 * np.array([-0.6])
XmolB     = -1 * np.array([7.])
R_outB    = np.array([200])
PAB       = np.array([135])
InclB     = np.array([30])
Pos_XB    = np.array([pos_Bx])
Pos_YB    = np.array([pos_By])
VsysB     = np.array([vsysB])
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Offset fitting values
# Parameters for Disk A
"""
TatmsA    = np.array([300])
TqqA      = -1 * np.array([-0.5])
XmolA     = -1 * np.array([8.])
R_outA    = np.array([300])
# PA and InclA are from Williams et al
PAA       = np.array([69.7])
InclA     = np.array([65])
Pos_XA    = np.array([pos_Ax])
Pos_YA    = np.array([pos_Ay])
VsysA     = np.array([vsysA])
# Pos_XA    = np.arange(pos_Ax - 0.05, pos_Ax + 0.15, 0.05)
# Pos_YA    = np.arange(pos_Ay - 0.05, pos_Ay + 0.15, 0.05)
# VsysA     = np.arange(vsysA - 0.11, vsysA + 0.1, 0.03)

# Parameters for Disk B
TatmsB    = np.array([400])
TqqB      = -1 * np.array([-0.5])
XmolB     = -1 * np.array([9.])
R_outB    = np.array([200])
PAB       = np.array([135])
InclB     = np.array([30])
# Pos_XB    = np.array([pos_Bx])
# Pos_YB    = np.array([pos_By])
# VsysB     = np.array([10.69])
Pos_XB    = np.arange(pos_Bx - 0.1, pos_Bx + 0.1, 0.05)
Pos_YB    = np.arange(pos_By - 0.1, pos_By + 0.2, 0.05)
VsysB     = np.arange(vsysB - 0.1, vsysB + 0.2, 0.05)
"""






diskAParams = np.array([TatmsA, TqqA, XmolA, R_outA, PAA, InclA, Pos_XA, Pos_YA, VsysA])
diskBParams = np.array([TatmsB, TqqB, XmolB, R_outB, PAB, InclB, Pos_XB, Pos_YB, VsysB])
