"""Add variances to the vis file.

Calculate the variance in a visibility map at each u,v point and each
channel. Calculate the variance based on the 70 nearest neighbors
"""

from astropy.io import fits
import numpy as np
import subprocess
import sys
import time


def var_vis(path_to_source):
    """Add variances to the vis file.

    Calculate the variance in a visibility map at each u,v point and each
    channel. Calculate the variance based on the 70 nearest neighbors
    """
    infile = path_to_source + '_exportuvfits.uvf'
    outfile = path_to_source + '.uvf'
    # read in u,v coordinates and visibilities
    im = fits.open('{}'.format(infile))

    # read in u,v coordinates in klam
    freq0 = im[0].header['crval4']
    klam = freq0/1e3
    u, v = im[0].data['UU'] * klam, im[0].data['VV'] * klam
    uvdist = np.sqrt(u**2 + v**2)

    nuv = u.size
    nfreq = 1

    # read in visiblities
    vis = (im[0].data['data']).squeeze()

    if vis.shape[1] == 2:  # polarized; turn to stokes
        real = (vis[:, 0, 0]+vis[:, 1, 0])/2.
        imag = (vis[:, 0, 1]+vis[:, 1, 1])/2.
    else:  # already stokes
        real = vis[:, :, 0]
        imag = vis[:, :, 1]

    print('')
    print(("{}: {} visibilities".format(infile, nuv)))

    # estimate good uvwidth using time-smearing equation
    # (eq. 3.194 in "Essential Radio Astronomy")
    synth_beam = 0.5                                            # arcsec
    # arcsec (originally at 5 from cail)
    angular_sep = 2.
    delta_t = synth_beam/angular_sep * 1/(2*np.pi)              # days

    # go from time to uv plane arc length
    baseline = np.median(np.sqrt(u**2 + v**2))
    delta_uv = delta_t * 2*np.pi*baseline

    # uvwidth: area around a particular uv point to consider when searching
    # for the nearest nclose neighbors.
    # (smaller numbers help make the run faster, but could result in many
    # points for which the weight cannot be calculated  and is left at 0)
    # make sure it's less than 1/10th of the shortest leg of the
    # rectangle defined by the longest u and v baseline
    uvwidth = delta_uv*2
    print(('uvwidth is {}'.format(uvwidth)))

    # nclose: number of nearby visibility points to use when measuring the dispersion
    # cut off at 220
    nclose = 50

    start = time.time()

    acceptance_ratio = 0

    going_up = False
    while True:
        print(('nclose is {}'.format(nclose)))

        real_weight = np.zeros((nuv, nfreq), dtype=np.float32)
        real_nclose_arr = np.zeros(len(u))
        imag_weight = np.zeros((nuv, nfreq), dtype=np.float32)
        imag_nclose_arr = np.zeros(len(u))
        # list of bad point indices used to calculate acceptance ratio
        bad_points = []

        for iuv in range(nuv):
            # Print counter inline
            print("IUV: ", iuv)
            sys.stdout.write('\033[F')

            # Boolean area (mask, basically) I think?
            w = (np.abs(u-u[iuv]) < uvwidth) & (np.abs(v-v[iuv]) < uvwidth)
            s = np.argsort(np.sqrt((v[w]-v[iuv])**2 + (u[w]-u[iuv])**2))

            # REALS
            # real_wf = (real[w, 0][s] != 0)
            real_wf = (real[w][s] != 0)
            real_nclose_arr[iuv] = real_wf.sum()
            if real_wf.sum() > nclose:
                # This is making all the channels the same.
                # Need two dimensions for looping.
                real_weight[iuv] = 1/np.std(real[w][s][real_wf][:nclose])**2
            else:
                # print iuv,real_wf.sum(),np.sqrt(u[iuv]**2+v[iuv]**2)
                print("Bad point added: ", iuv)
                bad_points.append(iuv)

            # IMAGINARIES
            imag_wf = (imag[w][s] != 0)
            imag_nclose_arr[iuv] = imag_wf.sum()
            if imag_wf.sum() > nclose:
                imag_weight[iuv] = 1/np.std(imag[w][s][imag_wf][:nclose])**2
            else:
                # print iuv,imag_wf.sum(),np.sqrt(u[iuv]**2+v[iuv]**2)
                if bad_points[-1] != iuv:
                    bad_points.append(iuv)

        acceptance_ratio = 1. - len(bad_points)/float(nuv)
        print("val: ", len(bad_points)/float(nuv))
        print("len(bad_points): ", len(bad_points))
        # print "float(nuv): ", float(nuv)
        print(("acceptance ratio: {}".format(acceptance_ratio)))

        if acceptance_ratio <= 0.99:
            nclose -= 10
            going_up = True
        # elif acceptance_ratio >= 0.99 and going_up == False:
        #    nclose += 10
        else:
            break

    # plt.plot(np.sqrt(u**2+v**2),nclose_arr,'.')
    # plt.scatter(real_weight, imag_weight)
    # plt.plot(imag_weight, imag_weight)
    # plt.xlabel("Real Weight")
    # plt.ylabel("Imaginary Weight")
    # plt.savefig("{}_weight_scatter.png".format(infile))

    # Calculating total weight from imaginary and real components
    total_weight = np.sqrt(real_weight*imag_weight)

    # Reshape weights to fit in weights column of uv fits file;
    # place the weights into both xx and yy polarization columns.
    if len(im[0].data['data'].shape) == 7:
        total_weight = np.reshape(total_weight, len(
            im[0].data['data'][:, 0, 0, 0, 0, 0, 2]))
        im[0].data['data'][:, 0, 0, 0, 0, 0, 2], im[0].data['data'][:,0, 0, 0, 0, 1, 2] = total_weight, total_weight
    else:
        total_weight = np.reshape(total_weight, len(
            im[0].data['data'][:, 0, 0, 0, 0, 2]))
        im[0].data['data'][:, 0, 0, 0, 0, 2], im[0].data['data'][:,0, 0, 0, 1, 2] = total_weight, total_weight

    # subprocess.call('rm {}.uvf'.format(outfile), shell=True)
    im.writeto('{}'.format(outfile))  # , overwrite=True)
    im.close()

    print(('Elapsed time (min): {}'.format((time.time()-start)/60.)))

    return real_weight, imag_weight

# Run it
# var_vis()


def create_vis(filename):
    subprocess.call('rm -r {}.vis'.format(filename), shell=True)
    subprocess.call('fits in={}.uvf op=uvin out={}.vis'.format(
        filename, filename), shell=True)


"""
uvfs = glob('unweighted_spws/*.uvf')
for uvf in uvfs:
    final_name = 'weighted_spws/' + uvf[16:30] + '_FINAL'
    var_vis(uvf[:-4], final_name)
    create_vis(final_name)
"""
