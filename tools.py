"""
Script some commands for easier calling, more specialized usage.

- Plot_fits: plot a fits image
- cgdisp: Display a map.
- imstat: Get rms and mean noise from a plane of the image.
- imspec: Display a spectrum.
- icr (invert/clean/restor): Convolve a model with the beam.
- sample_model_in_uvplane: convert a model fits to a sampled map (im, vis, uvf)
"""

# Packages
import os
import re
import argparse
import numpy as np
import subprocess as sp
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.pylab import *
from matplotlib.ticker import *
from matplotlib.pylab import figure
from matplotlib.patches import Ellipse as ellipse
from astropy.visualization import astropy_mpl_style
from constants import lines, get_data_path, obs_stuff, offsets, get_data_path, mol
plt.style.use(astropy_mpl_style)



def plot_fits(image_path, mol=mol, scale_cbar_to_mol=False,
              crop_arcsec=2, cmap='magma',
              save=True, show=True, use_cut_baselines=True):
    """
    Plot some fits image data.

    The cropping currently assumes a square image. That could be easily
    fixed by just adding y_center, y_min, and y_max and putting them in the
    imshow() call.
    Args:
        image_path (str): full path, including filetype, to image.
        crop_arcsec (float): How many arcseconds from 0 should the axis limits be set?
        nchans_to_cut (int): cut n/2 chans off the front and end
        cmap (str): colormap to use. Magma, copper, afmhot, CMRmap, CMRmap(_r) are nice

    Known Bugs:
        - Some values of n_chans_to_cut give a weird error. I don't really wanna
            figure that out right now

    To Do:
        - Maybe do horizontal layout for better screenshots for Evernote.
    """
    image_data = fits.getdata(image_path, ext=0).squeeze()
    header     = fits.getheader(image_path, ext=0)

    if scale_cbar_to_mol is True:
        # Get the data
        dataPath = get_data_path(mol, use_cut_baselines=True)
        real_data = fits.getdata(dataPath + '.fits', ext=0).squeeze()

        vmin = np.nanmin(real_data)
        vmax = np.nanmax(real_data)

    else:
        vmin = np.nanmin(image_data)
        vmax = np.nanmax(image_data)

    # Add some crosses to show where the disks should be centered (arcsec)
    # offsets_dA, offsets_dB = [-0.0298, 0.072], [-1.0456, -0.1879]
    offsets_dA, offsets_dB = offsets[0], offsets[1]

    # Beam stuff
    add_beam = True if 'bmaj' in header else False
    if add_beam is True:
        bmin = header['bmin'] * 3600.
        bmaj = header['bmaj'] * 3600.
        bpa = header['bpa']

    # Cropping: How many arcsecs radius should it be
    x_center = int(np.floor(image_data.shape[1]/2))
    # If zero is entered, just don't do any cropping and show full image.
    if crop_arcsec == 0:
        crop_arcsec = 256 * 0.045

    crop_pix = int(crop_arcsec / 0.045)
    xmin, xmax = x_center - crop_pix, x_center + crop_pix

    # Set up velocities:
    chanstep_vel = header['CDELT3'] * 1e-3
    # Reference pix value - chanstep * reference pix number
    chan0_vel = (header['CRVAL3'] * 1e-3) - header['CRPIX3'] * chanstep_vel

    # Cut 2/3 of nchans_to_cut out of the front end
    nchans_to_cut=12
    chan_offset = 2 * int(nchans_to_cut/3)
    nchans = image_data.shape[0] - nchans_to_cut

    # I think these are labeled backwards, but whatever.
    n_rows = int(np.floor(np.sqrt(nchans)))
    n_cols = int(np.ceil(np.sqrt(nchans)))
    chan_offset = 4

    fig = figure(figsize=[n_rows, n_cols])

    # Add the actual data
    for i in range(nchans):
        chan = i + int(np.floor(nchans_to_cut/2))
        velocity = str(round(chan0_vel + chan * chanstep_vel, 2))
        # i+1 because plt indexes from 1
        ax = fig.add_subplot(n_cols, n_rows, i+1)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(0, -0.8 * crop_arcsec, velocity + ' km/s',
                fontsize=6, color='w',
                horizontalalignment='center', verticalalignment='center')

        if i == n_rows * (n_cols - 1) and add_beam is True:
            el = ellipse(xy=[0.8 * crop_arcsec, 0.8*crop_arcsec],
                         width=bmin, height=bmaj, angle=-bpa,
                         fc='k', ec='w', fill=False, hatch='////////')
            ax.add_artist(el)

        cmaps = imshow(image_data[i + chan_offset][xmin:xmax, xmin:xmax],
                      cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=(crop_arcsec, -crop_arcsec,
                              crop_arcsec, -crop_arcsec))

        ax.plot(offsets_dA[0], offsets_dA[1], '+g')
        ax.plot(offsets_dB[0], offsets_dB[1], '+g')


    # Make the colorbar
    # Want it to be in the gap left at the end of the channels?
    inset_cbar = True
    if inset_cbar is True:
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        cax = plt.axes([0.55, 0.08, 0.38, 0.07])
        cbar = colorbar(cmaps, cax=cax, orientation='horizontal')
        cbar.set_label('Jy/beam',labelpad=-12,fontsize=12, weight='bold')
        cbar.set_ticks([vmin, vmax])

    # If not, put it on top.
    else:
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9)
        cax = plt.axes([0.1, 0.95, 0.81, 0.025])
        cbar = colorbar(cmaps, cax=cax, orientation='horizontal')
        cbar.set_label('Jy/beam',labelpad=-12,fontsize=12, weight='bold')
        cbar.set_ticks([vmin, vmax])


    # Want to save? Want to show?
    if save is True:
        if 'data' in image_path:
            resultsPath = 'data/' + mol + '/images/'
            if '-short' in image_path:
                suffix = '-' + image_path.split('-')[1].split('.')[0]
            else:
                suffix = ''
        elif 'mcmc' in image_path:
            resultsPath = 'mcmc_results/'
        elif 'gridSearch' in image_path:
            resultsPath = 'gridsearch_results'
        else:
            return 'Failed to make image; specify save location.'

        run_name = image_path.split('/')[-2]
        outpath = resultsPath + run_name + suffix + '_image.png'

        plt.savefig(outpath, dpi=200)
        print "Image saved to " + outpath

    if show is True:
        plt.show(block=False)
    plt.gca()


def cgdisp(imageName, crop=True, contours=True, olay=True, rms=6.8e-3):
    """Drop some sweet chanmaps.

    Args:
        imageName (str): name of the image to view, including .im or .cm
        crop (bool): whether or not to show constained image.
        contours (bool): whether or not to show 3-, 6-, and 9-sigma contours.
        rms (float): value to use for sigma.

    csize: 0 sets to default, and the third number controls 3pixel text size
    """
    if crop:
        r = '(-2,-2,2,2)'
    else:
        r = '(-5,-5,5,5)'

    call_str = ['cgdisp',
                'in={}'.format(imageName),
                'device=/xs',
                'region=arcsec,box{}'.format(r),
                'beamtyp=b,l,3',
                'labtyp=arcsec,arcsec,abskms',
                'options=3value',
                'csize=0,0.7,0,0'
                ]

    if contours:
        call_str[1] = 'in={},{}'.format(imageName, imageName)
        call_str.append('type=pix,con')
        call_str.append('slev=a,{}'.format(rms))
        call_str.append('levs1=3,6,9')
        call_str.append('options=3value,mirror,beambl')

    if olay:
        call_str.append('olay={}'.format('centering_for_olay.cgdisp'))

    sp.call(call_str)


def imstat(modelName, ext='.cm', plane_to_check=39):
    """Call imstat to find rms and mean.

    Want an offsource region so that we can look at the noise. Decision to look
    at plane_to_check=30 is deliberate and is specific to this line of these
    data. Look at June 27 notes for justification of it.
    Args:
        modelName (str): name of the input file. Not necessarily a model.
        plane_to_check (int): Basically which channel to look at, but that
        this includes the ~10 header planes, too, so plane 39 corresponds to
        channel 29 (v=11.4) or so.
    """
    print '\nIMSTATING ', modelName, '\n'

    r_offsource = '(-5,-5,5,-1)'
    imstat_raw = sp.check_output(['imstat',
                                  'in={}{}'.format(modelName, ext),
                                  'region=arcsec,box{}'.format(r_offsource)
                                  ])
    imstat_out = imstat_raw.split('\n')
    # Get column names
    hdr = filter(None, imstat_out[9].split(' '))

    # Split the output on spaces and then drop empty elements.
    imstat_list = filter(None, imstat_out[plane_to_check].split(' '))

    """Sometimes the formatting gets weird and two elements get squished
        together. Fix this with the split loop below.

        Note that the form of the elements of an imstat is 1.234E-05,
        sometimes with a '-' out front. Therefore, each the length of each
        positive element is <= 9 and <= 10 for the negative ones.
        """
    for i in range(len(imstat_list)-1):
        if len(imstat_list[i]) > 11:
            if imstat_list[i][0] == '-':
                cut = 10
            else:
                cut = 9
            imstat_list.insert(i+1, imstat_list[i][cut:])
            imstat_list[i] = imstat_list[i][:cut]

    d = {}
    for i in range(len(hdr) - 1):
        d[hdr[i]] = imstat_list[i]
        print hdr[i], ': ', imstat_list[i]

    # Return the mean and rms
    # return d
    return float(d['Mean']), float(d['rms'])


def icr(visPath, mol='hco', min_baseline=0, niters=1e4):
    """Invert/clean/restor: Turn a vis into a convolved clean map.

    .vis -> .bm, .mp, .cl, .cm, .fits
    Args:
        modelName (str): path to and name of the file. Do not include extension
        min_baseline (int): minimum baseline length to use. Cuts all below.
        niters (int): how many iterations of cleaning to run. Want this to be
                      big enough that it never gets triggered.
        rms (float): the the rms noise, to which we clean down to.
        mol (str): which molecule's restfreq to use.
    """
    print "\nConvolving image\n"

    # Since the path is sometimes more than 64 characters long, need to slice
    # it and use Popen/cwd to cut things down.
    # filepath = visPath.split('/')[1:-1]
    if '/' in visPath:
        visName = visPath.split('/')[-1]
        filepath = visPath[:-len(visName)]
    else:
        visName = visPath
        filepath = './'

    # Add a shorthand name (easier to write)
    # Rename the outfile if we're cutting baselines and add the cut call.
    b = '' if min_baseline == 0 else min_baseline
    outName = visName + str(b)

    # Add restfreq to this vis
    rf = lines[mol]['restfreq']
    sp.Popen(['puthd',
              'in={}.vis/restfreq'.format(visName),
              'value={}'.format(rf)],
             stdout=open(os.devnull, 'wb'),
             cwd=filepath).wait()

    for end in ['.cm', '.cl', '.bm', '.mp']:
        remove(outName + end)
        # sp.Popen('rm -rf {}.{}'.format(outName, end), shell=True, cwd=filepath).wait()

    # Invert stuff:
    invert_str = ['invert',
                  'vis={}.vis'.format(visName),
                  'map={}.mp'.format(outName),
                  'beam={}.bm'.format(outName),
                  'options=systemp',
                  'cell=0.045',
                  'imsize=256',
                  'robust=2']

    if min_baseline != 0:
        invert_str.append('select=-uvrange(0,{})'.format(b))

    # sp.call(invert_str, stdout=open(os.devnull, 'wb'))
    sp.Popen(invert_str, stdout=open(os.devnull, 'wb'), cwd=filepath).wait()

    # Grab the rms
    rms = imstat(filepath + outName, '.mp')[1]

    sp.Popen(['clean',
              'map={}.mp'.format(outName),
              'beam={}.bm'.format(outName),
              'out={}.cl'.format(outName),
              'niters={}'.format(niters),
              'threshold={}'.format(rms)],
             # stdout=open(os.devnull, 'wb')
             cwd=filepath).wait()

    sp.Popen(['restor',
              'map={}.mp'.format(outName),
              'beam={}.bm'.format(outName),
              'model={}.cl'.format(outName),
              'out={}.cm'.format(outName)],
             stdout=open(os.devnull, 'wb'),
             cwd=filepath).wait()

    sp.Popen(['fits',
              'op=xyout',
              'in={}.cm'.format(outName),
              'out={}.fits'.format(outName)],
             cwd=filepath).wait()


def sample_model_in_uvplane(modelPath, mol='hco', option='replace'):
    """Sample a model image in the uvplane given by the data.

    .fits -> {.im, .uvf, .vis}
    Args:
        modelPath (str): path to model fits file.
        dataPath (str): path to data vis file.
        mol (str): the molecule we're looking at.
        option (str): Choose whether we want a simple sampling (replace),
                      or a residual (subtract).
    """
    dataPath = get_data_path(mol)
    # Oooooo baby this is janky. Basically just wanting to have the name
    # reflect that it's a residual map if that's what's chosen.
    old_modelPath = modelPath
    modelPath = modelPath + '_resid' if option == 'subtract' else modelPath

    if option == 'subtract':
        print "Making residual map."

    remove(modelPath + '.im')
    sp.call(['fits', 'op=xyin',
             'in={}.fits'.format(old_modelPath),
             'out={}.im'.format(modelPath)])

    # Sample the model image using the observation uv coverage
    remove(modelPath + '.vis')
    sp.call(['uvmodel',
             'options={}'.format(option),
             'vis={}.vis'.format(dataPath),
             'model={}.im'.format(modelPath),
             'out={}.vis'.format(modelPath)])

    # Convert to UVfits
    remove(modelPath + '.uvf')
    sp.call(['fits',
             'op=uvout',
             'in={}.vis'.format(modelPath),
             'out={}.uvf'.format(modelPath)])

    print "completed sampling uvplane; created .im, .vis, .uvf\n\n"


def imspec(imageName, source='both'):
    """Drop a sweet spectrum. Takes in a .im."""

    if source.lower() == 'both':
        r = '(-2, -2, 2, 2)'
    elif source.lower() == 'a':
        r = '(-0.8,-0.8,1,0.8)'
    elif source.lower() == 'b':
        r = '(-1.6,-1.,-0.7,1)'
    else:
        return "Choose A, B, or Both"

    sp.call(['imspec',
             'in={}'.format(imageName),
             'region=arcsec,box{}'.format(r),
             'device=/xs, plot=sum'])


def already_exists(query):
    """Search an ls call to see if query is in it.

    The loop here is to check that directories in the path of the query also
    exist.
    """
    f = query.split('/')
    f = filter(None, f)
    f = f[1:] if f[0] == '.' else f

    i = 0
    while i < len(f):
        short_path = '/'.join(f[:i])
        if short_path == '':
            output = sp.check_output('ls').split('\n')
        else:
            output = sp.check_output('ls', cwd=short_path).split('\n')

        if f[i] not in output:
            #print "False"
            #break
            return False
        else:
            i += 1
    # print "True"
    return True


def already_exists_old(query):
    """Search an ls call to see if query is in it."""
    f = query.split('/')[-1]
    # path = query.split(f)[0]
    path = query[:-len(f)]

    # print "Path is: ", path
    # print "file is: ", f

    output = sp.check_output('ls', cwd=path).split('\n')

    if f in output:
        print query + ' alrady exists; skipping\n'
        return True
    else:
        print query + ' does not yet exist; executing command\n'
        return False


def remove(filePath):
    """Delete some files.

    Mostly just written to avoid having to remember the syntax every time.
    filePath is full filepath, including name and extension.
    Supports wildcards.
    """
    if filePath == '/':
        return "DONT DO IT! DONT BURN IT ALL DOWN YOU STOOP!"

    if type(filePath) == str:
        sp.Popen(['rm -rf {}'.format(filePath)], shell=True).wait()

    elif type(filePath) == list:
        for f in filePath:
            sp.Popen(['rm -rf {}'.format(f)], shell=True).wait()


def pipe(commands):
    """Translate a set of arguments into a CASA command, written by Cail."""
    call_string = '\n'.join([command if type(command) is str else '\n'.join(command) for command in commands])

    print('Piping the following commands to CASA:\n')
    print(call_string)
    # sp.call(['casa', '-c', call_string])
    # sp.Popen(['casa', '-c', call_string]).wait()
    sp.Popen(['casa', '--nologger', '-c', call_string]).wait()
    # sp.Popen(['casa', '--nologger', '--log2term', '-c', call_string]).wait()

    # clean up .log files that casa poops out
    sp.Popen('rm -rf *.log', shell=True).wait()


def tclean(mol='hco', output_path='./test'):
    """Fix SPW - it's a guess.

    What is phasecenter?
    Which weighting should I use?
    What robustness level?
    What restoringbeam?
    """
    chan_min = str(lines[mol]['chan0'])
    # chan_step = obs_stuff('hco')[4]
    restfreq = lines[mol]['restfreq']
    hco_im = fits.getheader('./data/hco/hco.fits')
    hco_vis = fits.getheader('./data/hco/hco.uvf')
    chan_step = hco_vis['CDELT4'] * 1e-9
    pipe(["tclean(",
          "vis       = '../../raw_data/calibrated-{}.ms.contsub',".format(mol),
          "imagename     = './tclean_test',",
          "field         = 'OrionField4',",
          "spw           = '',",
          "specmode      = 'cube',",
          # "nchan         = 51,",
          # "start         = '{}GHz',".format(chan_min),
          # "width         = '{}GHz',".format(chan_step),
          "outframe      = 'LSRK',",
          "restfreq      = '{}GHz',".format(restfreq),
          "deconvolver   = 'hogbom',",
          "gridder       = 'standard',",
          "imsize        = [256, 256],",
          "cell          = '0.045arcsec',",
          "interactive   = False,",
          "niter         = 5000)"])









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Do the tools.')
    parser.add_argument('-t', '--tclean', action='store_true',
                        help='Run a tcleaning.')
    args = parser.parse_args()
    if args.tclean:
        tclean('hco')

# The End
