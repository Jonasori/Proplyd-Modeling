"""Data processing pipeline.

A small change
To be run from /Volumes/disks/jonas/freshStart/modeling
sp.Popen vs sp.call: Popen lets you change working directory for the call with
                     cwd arg, call blocks further action until its action is
                     complete, although Popen can be made to block with .wait()
"""


import time
import argparse
import numpy as np
import subprocess as sp
from astropy.io import fits
from astropy.constants import c
from constants import lines, today
from var_vis import var_vis
from tools import icr, already_exists, pipe, remove
c = c.to('km/s').value

def casa_sequence(mol, raw_data_path, output_path,
                  cut_baselines=False, remake_all=False):
    """Cvel, split, and export as uvf the original cont-sub'ed .ms.

    Args:
        - mol:
        - raw_data_path:
        - output_path: path, with name included
        - spwID: the line's spectral window ID. spwID(HCO+) = 1
        - cut_baselines: obvious.
        - remake_all (bool): if True, remove delete all pre-existing files.
    """
    # FIELD SPLIT
    remove(raw_data_path + '-' + mol + '.ms')
    pipe(["split(",
          "vis='{}calibrated.ms',".format(raw_data_path),
          "outputvis='{}calibrated-{}.ms',".format(raw_data_path, mol),
          "field='OrionField4',",
          "spw={})".format(lines[mol]['spwID'])
          ])

    # CONTINUUM SUBTRACTION
    # Want to exlude the data disk from our contsub, so use split_range
    # By the time this gets used there is only one spw so 0 is fine
    split_range = find_split_cutoffs(mol)
    spw = '0:' + str(split_range[0]) + '~' + str(split_range[1])
    remove(raw_data_path + '-' + mol + '.ms.contsub')
    pipe(["uvcontsub(",
          "vis='{}calibrated-{}.ms',".format(raw_data_path, mol),
          "fitspw='{}',".format(spw),
          "excludechans=True,",
          "spw='0')"])

    # CVEL
    remove(output_path + '_cvel.ms')
    chan0_freq    = lines[mol]['chan0_freq']
    chanstep_freq = lines[mol]['chanstep_freq']
    restfreq      = lines[mol]['restfreq']
    chan0_vel     = c * (chan0_freq - restfreq)/restfreq
    chanstep_vel  = c * (chanstep_freq/restfreq)
    pipe(["cvel(",
          "vis='{}calibrated-{}.ms.contsub',".format(raw_data_path, mol),
          "outputvis='{}_cvel.ms',".format(output_path),
          "field='',",
          "mode='velocity',",
          "nchan=-1,",
          "width='{}km/s',".format(chanstep_vel),
          "start='{}km/s',".format(chan0_vel),
          "restfreq='{}GHz',".format(lines[mol]['restfreq']),
          "outframe='LSRK')"
          ])

    # SPLIT OUT VALUABLE CHANNELS
    # Using the choices made earlier, split out the channels we want
    # I'm concerned about the order of this; seems like it the desired split
    # range will change before and after cvel
    remove(output_path + '_split.ms')
    split_str = (["split(",
                  "vis='{}_cvel.ms',".format(output_path),
                  "outputvis='{}_split.ms',".format(output_path),
                  "spw='{}',".format(spw),
                  "datacolumn='all',",
                  "keepflags=False)"
                  ])

    # If necessary, insert a baseline cutoff. Because we want
    # to keep the ) in the right spot, just put uvrange= in the middle.
    if cut_baselines is True:
        print "\nCutting baselines in casa_sequence\n"
        b_min = lines[mol]['baseline_cutoff']
        split_str = split_str[:-2] + \
            [("uvrange='>" + str(b_min) + "klambda',")] + \
            split_str[-2:]

    pipe(split_str)

    # EXPORT IT
    remove(output_path + '_exportuvfits.uvf')
    pipe(["exportuvfits(",
          "vis='{}_split.ms',".format(output_path),
          "fitsfile='{}_exportuvfits.uvf')".format(output_path)
          ])


def find_split_cutoffs(mol, other_restfreq=0):
    """Find the indices of the 50 channels around the restfreq.

    chan_dir, chan0_freq, nchans, chanstep from
    listobs(vis='raw_data/calibrated-mol.ms.contsub')
    """
    # ALl in GHz. Both values pulled from listobs
    chan0_freq = lines[mol]['chan0_freq']
    chanstep = lines[mol]['chanstep_freq']
    restfreq = lines[mol]['restfreq']
    nchans = 3840

    freqs = [chan0_freq + chanstep*i for i in range(nchans)]

    # Find the index of the restfreq channel
    loc = 0
    min_diff = 1
    for i in range(len(freqs)):
        diff = abs(freqs[i] - restfreq)
        if diff < min_diff:
            min_diff = diff
            loc = i

    # Need to account for the systemic velocitys shift. Do so by rearranging
    # d_nu/nu = dv/c
    # ((sysv/c) * restfreq)/chanstep = nchans of shift to apply
    # = (10.55/c) * 356.734223/0.000488281 = 25.692
    shift = int(10.55/c * abs(restfreq/chanstep))
    # So shift in an extra 26 channels
    loc = loc - shift if loc > shift else -np.inf
    split_range = [loc - shift, loc + shift]

    return split_range


def baseline_cutter(mol):
    """Cut a vis file.

    It seems like doing this with uvaver is no good because it drops the
    SPECSYS keyword from the header, so now implementing it with CASA in the
    split early on, so this is no longer used.

    This one uses Popen and cwd (change working directory) because the path was
    getting to be longer than buffer's 64-character limit. Could be translated
    to other funcs as well, but would just take some work.
    """
    filepath = './data/' + mol + '/'
    min_baseline = lines[mol]['baseline_cutoff']
    name = mol
    new_name = name + '-short' + str(min_baseline)

    print "\nCompleted uvaver; starting fits uvout\n"
    sp.call(['fits',
             'op=uvout',
             'in={}.vis'.format(new_name),
             'out={}.uvf'.format(new_name)],
            cwd=filepath)

    # Now clean that out file.
    print "\nCompleted fits uvout; starting ICR\n\n"
    icr(filepath + new_name, mol)

    # For some reason icr is returning and so it never deletes these. Fix later
    sp.Popen(['rm -rf {}.bm'.format(new_name)], shell=True)
    sp.Popen(['rm -rf {}.cl'.format(new_name)], shell=True)
    sp.Popen(['rm -rf {}.mp'.format(new_name)], shell=True)


def run_full_pipeline():
    """Run the whole thing.

    Note that this no longer produces both cut and uncut output; since the cut
    happens much earlier, it now only produces one or the other (depending
    on whether or not cut_baselines is true.)
    The Process:
        - casa_sequence():
            - cvel the cont-sub'ed dataset from jonas/raw_data to here.
            - split out the 50 channels around restfreq
            - convert that .ms to a .uvf
        - var_vis(): pull in that .uvf, add variances, resulting in another uvf
        - convert that to a vis
        - icr that vis to get a cm
        - cm to fits; now we have mol.{{uvf, vis, fits, cm}}
        - delete the clutter files: _split, _cvel, _exportuvfits, bm, cl, mp
    """
    t0 = time.time()
    mol = raw_input('Which line (HCN, HCO, CS, or CO)?\n').lower()
    cut = raw_input('Cut baselines for better signal (y/n)?\n').lower()
    cut_baselines = True if cut == 'y' else False
    remake = raw_input('Remake everything (y/n)?\n')
    remake_all = True if remake.lower() == 'y' else False

    # Paths to the data
    jonas = '/Volumes/disks/jonas/'
    raw_data_path = jonas + 'raw_data/'
    final_data_path = jonas + 'modeling/data/' + mol + '/'
    name = mol
    if cut_baselines is True:
        name += '-short' + str(lines[mol]['baseline_cutoff'])

    # Establish a string for the log file to be made at the end
    log = 'Files created on ' + today + '\n\n'

    if remake_all is True:
        # This doesn't work yet.
        print "Remaking everything; emptied line dir and remaking."
        remove(final_data_path + '*')
        log += "Full remake occured; all files are fresh.\n\n"
    else:
        log += "Some files already existed and so were not remade.\n"
        log += "Careful for inconsistencies.\n\n"

    print "Now processing data...."
    casa_sequence(mol, raw_data_path,
                  final_data_path + name, cut_baselines)

    print "Running varvis....\n\n"
    if already_exists(final_data_path + name + '.uvf') is False:
        # Note that var_vis takes in mol_exportuvfits, returns mol.uvf
        var_vis(final_data_path + name)
    print "Finished varvis; converting uvf to vis now....\n\n"

    # Note that this is different than lines[mol][chan0_freq] bc
    # it's dealing with the chopped vis set
    restfreq = lines[mol]['restfreq']
    f = fits.getheader(final_data_path + name + '.uvf')

    # chan0_freq = (f['CRVAL4'] - (f['CRPIX4']-1) * f['CDELT4']) * 1e-9
    # Using the same math as in lines 130-135
    # chan0_vel = c * (chan0_freq - restfreq)/restfreq
    data, header = fits.getdata(final_data_path + name + '.uvf', header=True)
    header['RESTFREQ'] = restfreq * 1e9
    fits.writeto(final_data_path + name + '.uvf', data, header, overwrite=True)
    if already_exists(final_data_path + name + '.vis') is False:
        sp.Popen(['fits',
                  'op=uvin',
                  'in={}.uvf'.format(name),
                  # DONT PUT THIS BACK IN
                  # Or if you do, flip the sign of chan0_vel to pos
                  # 'velocity=lsr,{},1'.format(chan0_vel),
                  'out={}.vis'.format(name)],
                 cwd=final_data_path).wait()


    print "Convolving data to get image, converting output to .fits\n\n"
    if already_exists(final_data_path + name + '.cm') is False:
        icr(final_data_path + name, mol=mol)

    print "Deleting the junk process files...\n\n"
    fpath = final_data_path + name
    files_to_remove = [fpath + '.bm', fpath + '_split.*',
                       fpath + '.cl', fpath + '_cvel.*',
                       fpath + '.mp', fpath + '_exportuvfits.*',
                       'casa*.log', '*.last']
    remove(files_to_remove)

    tf = time.time()
    t_total = (tf - t0)/60
    log += '\nThis processing took ' + str(t_total) + ' minutes.'
    with open(final_data_path + 'file_log.txt', 'w') as f:
        f.write(log)
    print "All done! This processing took " + str(t_total) + " minutes."


# ARGPARSE STUFF
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the data.')
    parser.add_argument('-r', '--run', action='store_true',
                        help='Run the processor.')
    args = parser.parse_args()
    if args.run:
        run_full_pipeline()










# The End
