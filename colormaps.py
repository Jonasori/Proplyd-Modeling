"""Some colormaps."""

import seaborn as sns
import numpy as np


def make_cmap(colors, position=None, bit=False):
    """Docstring."""
    import matplotlib as mpl
    bit_rgb = np.linspace(0, 1, 256)
    if position == None:
        position = np.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            system.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            system.exit('position must start with 0 and end with 1')
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return cmap


colors = [(255, 255, 255), (240, 255, 255),
          (210, 253, 255), (184, 252, 255), (192, 244, 204), (155, 255, 145),
          (210, 200, 12), (230, 180, 7), (236, 124, 13), (233, 100, 25),
          (228, 30, 45), (198, 0, 46), (103, 0, 51)]
jesse_reds = make_cmap(colors, bit=True)

cubehelix_1 = sns.cubehelix_palette(n_colors=2,
                                    start=3,
                                    rot=1,
                                    hue=2,
                                    light=1,
                                    dark=0.1,
                                    as_cmap=True)

cubehelix_2 = sns.cubehelix_palette(start=0,
                                    rot=1,
                                    hue=2,
                                    gamma=0.6,
                                    light=1,
                                    dark=0,
                                    as_cmap=True)

cubehelix_3 = sns.cubehelix_palette(start=0,
                                    rot=1,
                                    gamma=.8,
                                    hue=2,
                                    light=1,
                                    dark=0,
                                    as_cmap=True)
