"""
Organize the different line observations.

Very much outdated.
"""

from fitting import Observation
import numpy as np
from constants import lines

hco = Observation(root='./data/hco/',
                       name='hco',
                       rms=1e-3,
                       restfreq=lines['hco']['restfreq'])

"""
hcn = Observation(root='./data/hcn/',
                  name='hcn',
                  rms=1e-3,
                  restfreq=lines['hco']['restfreq'])

co = Observation(root='./data/co/',
                 name='co',
                 rms=1e-3,
                 restfreq=lines['hco']['restfreq'])

cs = Observation(root='./data/cs/',
                 name='cs',
                 rms=1e-3,
                 restfreq=lines['hco']['restfreq'])
"""
# A list of those observations
# observations = np.array([hco, hcn, co, cs])
observations = np.array([hco])

# Maybe it'd be nicer to have it as a dict

observations_dict = {'hco': hco}
#                     'hcn': hcn,
#                     'co': co,
#                     'cs': cs
#                     }


lines = {'hco': {'obs': hco, 'restfreq': 356.73422300, 'jnum': 3, 'rms': 1}}
"""
	 'hcn': {'obs': hcn, 'restfreq': 354.50547590, 'jnum': 3, 'rms': 1},
	 'co': {'obs': co, 'restfreq': 345.79598990, 'jnum': 3, 'rms': 1},
	 'cs': {'obs': cs, 'restfreq': 342.88285030, 'jnum': 3, 'rms': 1}
	 }
"""


# The End
