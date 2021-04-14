import copy
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime


plt.rcParams.update({'font.size': 12})


def visualize_wave_conditions(df, title=None):
    df = copy.deepcopy(df)
    df.t = df.t.apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
    t = df.t.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    fig, axes = plt.subplots(4, 1, figsize=(13, 7.5))
    axes[0].plot(t, df.hs)
    axes[0].set_ylabel('Hs [m]')
    axes[1].plot(t, df.tp)
    axes[1].set_ylabel('Tp [s]')
    axes[2].plot(t, df.dir)
    axes[2].set_ylabel('Dir [degrees]')
    axes[3].plot(t, (df.hs ** 2) * df.tp)
    axes[3].set_ylabel('Wave power ((Hs^2)*Tp)')

    if title:
        plt.suptitle(title)
    plt.show()


def visualize_topo_2d_1d(topo, vmin=None, vmax=None, title=None):
    vmin = np.min(topo.df.z) if vmin is None else vmin
    vmax = np.max(topo.df.z) if vmax is None else vmax

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im = axes[0].scatter(topo.df.lng, topo.df.lat, c=topo.df.z, marker='.', vmin=vmin, vmax=vmax)
    axes[0].set_title('Raw survey')
    axes[0].xaxis.set_tick_params(rotation=-45)
    plt.colorbar(im, ax=axes[0], shrink=0.7)

    avg_df = topo.get_average_cross_shore_profile()
    axes[1].plot(avg_df.lng, avg_df.z)
    axes[1].set_title('Cross-shore average')
    axes[1].xaxis.set_tick_params(rotation=-45)
    axes[1].ticklabel_format(useOffset=False)

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()
