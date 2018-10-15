from astropy.io import fits
from astropy.modeling import models, fitting
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse
from matplotlib.ticker import MultipleLocator, LinearLocator, AutoMinorLocator
import colormaps
import matplotlib.patheffects as PathEffects
import sklearn.neighbors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Figure:
    """Docstring."""

    # Set seaborn plot styles and color pallete
    sns.set_style("ticks",
                  {"xtick.direction": "in",
                   "ytick.direction": "in"})
    sns.set_context("talk")

    def __init__(self, paths, rmses, texts, layout=(1, 1),  savefile='figure.pdf', title=None, show=False):
        """Docstring."""
        rmses = np.array([rmses]) if type(rmses) is float else np.array(rmses)
        texts = np.array([texts], dtype=object) if type(texts) is str \
            else np.array(texts, dtype=object)
        paths = np.array([paths]) if type(paths) is str else np.array(paths)
        self.title = title
        self.rows, self.columns = layout

        # clear any pre existing figures, then create figure
        plt.close()
        self.fig, self.axes = plt.subplots(self.rows, self.columns,
                                           figsize=(
                                               11.6/2 * self.columns, 6.5*self.rows),
                                           sharex=False, sharey=False, squeeze=False)
        plt.subplots_adjust(wspace=-0.0)

        if type(texts.flatten()[0]) is not float:
            texts = texts.flatten()
        for ax, path, rms, text in zip(self.axes.flatten(), paths.flatten(), rmses.flatten(), texts):
            self.rms = rms
            self.get_fits(path)
            self.make_axis(ax)
            self.fill_axis(ax, text)

        if savefile:
            plt.savefig(savefile, dpi=700)
        if show:
            plt.show()

    def get_fits(self, path):
        """Docstring."""
        fits_file = fits.open(path)
        self.head = fits_file[0].header
        self.im = fits_file[0].data[0][0]
        # self.im[np.isnan(self.im)]=0.

        # change units to micro Jy
        self.im *= 1e6
        self.rms *= 1e6

        # Read in header spatial info to create ra
        nx = self.head['NAXIS1']
        ny = self.head['NAXIS2']
        xpix = self.head['CRPIX1']
        ypix = self.head['CRPIX2']
        xval = self.head['CRVAL1']
        yval = self.head['CRVAL2']
        self.xdelt = self.head['CDELT1']
        self.ydelt = self.head['CDELT2']

        # Convert from degrees to arcsecs
        self.ra_offset = np.array(
            ((np.arange(nx) - xpix + 1) * self.xdelt) * 3600)
        self.dec_offset = np.array(
            ((np.arange(ny) - ypix + 1) * self.ydelt) * 3600)

    def make_axis(self, ax):
        """Docstring."""
        # Set seaborn plot styles and color pallete
        sns.set_style("ticks",
                      {"xtick.direction": "in",
                       "ytick.direction": "in"})
        sns.set_context("talk")

        xmin = -5.0
        xmax = 5.0
        ymin = -5.0
        ymax = 5.0
        ax.set_xlim(xmax, xmin)
        ax.set_ylim(ymin, ymax)
        ax.grid(False)

        # Set x and y major and minor tics
        majorLocator = MultipleLocator(1)
        ax.xaxis.set_major_locator(majorLocator)
        ax.yaxis.set_major_locator(majorLocator)

        minorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_minor_locator(minorLocator)

        # Set x and y labels
        ax.set_xlabel(r'$\Delta \alpha$ (")', fontsize=18)
        ax.set_ylabel(r'$\Delta \delta$ (")', fontsize=18)
        ax.xaxis.set_ticklabels(
            ['', '', '-4', '', '-2', '', '0', '', '2', '', '4', ''], fontsize=18)
        ax.yaxis.set_ticklabels(
            ['', '', '-4', '', '-2', '', '0', '', '2', '', '4', ''], fontsize=18)
        ax.tick_params(which='both', right='on', labelsize=18, direction='in')

        # Set labels depending on position in figure
        if np.where(self.axes == ax)[1] % self.columns == 0:  # left
            ax.tick_params(axis='y', labelright='off', right='on')
        elif np.where(self.axes == ax)[1] % self.columns == self.columns - 1:  # right
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft='off', labelright='on')
        else:  # middle
            ax.tick_params(axis='y', labelleft='off')
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Set physical range of colour map
        self.extent = [self.ra_offset[0], self.ra_offset[-1],
                       self.dec_offset[-1], self.dec_offset[0]]

    def fill_axis(self, ax, text):
        """Docstring."""
        # Plot image as a colour map
        cmap = ax.imshow(self.im,
                         extent=self.extent,
                         vmin=np.min(self.im),
                         vmax=np.max(self.im),
                         cmap=colormaps.jesse_reds)

        if self.rms:
            # Set contour levels
            cont_levs = np.arange(3, 100, 3) * self.rms
            # add residual contours if resdiual exists; otherwise, add image contours
            try:
                ax.contour(self.resid,
                           levels=cont_levs,
                           colors='k',
                           linewidths=0.75,
                           linestyles='solid')
                ax.contour(self.resid,
                           levels=-1 * np.flip(cont_levs, axis=0),
                           colors='k',
                           linewidths=0.75,
                           linestyles='dashed')
            except AttributeError:
                ax.contour(self.ra_offset, self.dec_offset, self.im,
                           colors='k',
                           levels=cont_levs,
                           linewidths=0.75,
                           linestyles='solid')
                ax.contour(self.ra_offset, self.dec_offset, self.im,
                           levels=-1 * np.flip(cont_levs, axis=0),
                           colors='k',
                           linewidths=0.75,
                           linestyles='dashed')

        # Create the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="8%", pad=0.0)
        cbar = self.fig.colorbar(cmap, ax=ax, cax=cax,
                                 orientation='horizontal')
        cbar.ax.xaxis.set_tick_params(direction='out', length=3, which='major',
                                      bottom='off', top='on', labelsize=8, pad=-2,
                                      labeltop='on', labelbottom='off')

        cbar.ax.xaxis.set_tick_params(direction='out', length=2, which='minor',
                                      bottom='off', top='on')

        if np.max(self.im) > 500:
            tickmaj = 200
            tickmin = 50
        elif np.max(self.im) > 200:
            tickmaj = 100
            tickmin = 25
        elif np.max(self.im) > 100:
            tickmaj = 50
            tickmin = 10
        elif np.max(self.im) <= 100:
            tickmaj = 20
            tickmin = 5

        minorLocator = AutoMinorLocator(tickmaj / tickmin)
        cbar.ax.xaxis.set_minor_locator(minorLocator)
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(),
                                rotation=45, fontsize=18)
        cbar.set_ticks(np.arange(-10*tickmaj, 10*tickmaj, tickmaj))

        # Colorbar label
        cbar.ax.text(0.425, 0.320, r'$\mu Jy / beam$', fontsize=12,
                     path_effects=[PathEffects.withStroke(linewidth=2, foreground="w")])

        # Overplot the beam ellipse
        try:
            beam_ellipse_color = 'k'
            bmin = self.head['bmin'] * 3600.
            bmaj = self.head['bmaj'] * 3600.
            bpa = self.head['bpa']

            el = Ellipse(xy=[4.2, -4.2], width=bmin, height=bmaj, angle=-bpa,
                         edgecolor='k', hatch='///', facecolor='none', zorder=10)
            ax.add_artist(el)
        except KeyError:
            pass

        # Plot the scale bar
        if np.where(self.axes == ax)[1][0] == 0:  # if first plot
            x = -3.015
            y = -4.7
            ax.plot(
                [x, x - 10/9.725],
                [y, y],
                '-', linewidth=2, color='k')
            ax.text(
                x + 0.32, y + 0.15, "10 au",
                fontsize=18,
                path_effects=[PathEffects.withStroke(linewidth=2, foreground="w")])

        # Plot a cross at the source position
        # ax.plot([0.0], [0.0], '+', markersize=6, markeredgewidth=1, color='w')

        # Add figure text
        if text is not None:
            for t in text:
                ax.text(*t, fontsize=18,
                        path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])

        if self.title:
            plt.suptitle(self.title)

    def quickview(self):
        plt.imshow(self.im, origin='lower')
        plt.show(block=False)


def my_kde(samples, ax=None, show=False, **kwargs):
    """Docstring.""" 
    # if ax is None: fig, ax = plt.subplots()

    q1, q3 = samples.quantile([.25, .75])
    scotts = 1.059 * min(samples.std(), q3-q1) * samples.size ** (-1 / 5.)
    x_grid = np.linspace(samples.min(), samples.max(), 500)

    kde = sklearn.neighbors.KernelDensity(bandwidth=2*scotts)
    kde.fit(samples.values.reshape(-1, 1))
    pdf = np.exp(kde.score_samples(x_grid.reshape(-1, 1)))

    ax.plot(x_grid, pdf)

    if show:
        plt.show()

    # if using this line, make sure to import sklearn.model_selection
    # bw_search = sklearn.model_selection.GridSearchCV(
    #     sklearn.neighbors.kde.KernelDensity(),
    #     {'bandwidth': scotts * np.array([0.1, 0.5, 1, 2, 4, 5, 10])})


# attempt at making kde plot
#=========================================================================
# run_name = 'run5_26walkers_10params'
# posterior = pd.read_csv(run_name + '.csv')
#
#
# multi = posterior[['m_disk', 'sb_law']]
#
# sns.kdeplot(multi)
# plt.show()
#
# bw_search = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0, multi.std().mean()/10, 5)}, cv=20)
# bw_search.fit(multi)
# multi.shape[0]**(-1./(multi.shape[1]+4))
# multi.std()
# **()
#
# xx, yy = np.meshgrid(np.linspace(*multi.iloc[:,0].quantile([0,1]), num=100),
#                      np.linspace(*multi.iloc[:,1].quantile([0,1]), num=100))
# test = np.array([xx.ravel(), yy.ravel()]).T
# kde=KernelDensity(bandwidth=multi.std().min()/10)
# kde.fit(multi)
# pdf = np.exp(kde.score_samples(multi)).reshape(len(xx), -1)
#
# plt.contour(xx, yy, pdf )
# plt.savefig('test.png')
# plt.show()
#
#
#
