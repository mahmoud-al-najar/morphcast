import os
import copy
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate
from itertools import groupby

from utilities.wrappers import Topo
from utilities.common import get_datetime_from_ymd_string, datetime_to_timestamp


data_dir = '/media/mn/WD4TB/topo/survey_dems/data'

root_out_dir = '/media/mn/WD4TB/topo/survey_dems/processed'
out_grids_dir_path = os.path.join(root_out_dir, 'grids')
out_plots_dir_path = os.path.join(root_out_dir, 'plots')
out_xyz_dir_path = os.path.join(root_out_dir, 'xyz')

files = os.listdir(data_dir)
n_files = len(files)
count = 0

for f in files:
    out_file = f.replace('.nc', '').replace('FRF_geomorphology_DEMs_surveyDEM_', '')

    out_grid_path = os.path.join(out_grids_dir_path, out_file + '.npy')
    out_plot_path = os.path.join(out_plots_dir_path, out_file + '.png')
    out_xyz_path = os.path.join(out_xyz_dir_path, out_file + '.xyz')

    if (not os.path.isfile(out_grid_path)) or \
            (not os.path.isfile(out_plot_path)) or \
            (not os.path.isfile(out_xyz_path)):
        ncd = netCDF4.Dataset(os.path.join(data_dir, f))
        time = ncd.variables['time']
        x_frf = ncd.variables['xFRF']
        y_frf = ncd.variables['yFRF']
        project = ncd.variables['project']

        list_lng = []
        list_lat = []
        list_elevation = []
        for i_t in range(len(time)):
            for i_x in range(len(x_frf)):
                for i_y in range(len(y_frf)):
                    lng = ncd.variables['longitude'][i_y, i_x]
                    lat = ncd.variables['latitude'][i_y, i_x]
                    elevation = ncd.variables['elevation'][i_t, i_y, i_x]
                    list_lng.append(lng)
                    list_lat.append(lat)
                    list_elevation.append(elevation)

        df = pd.DataFrame({'lng': np.array(list_lng),
                           'lat': np.array(list_lat),
                           'z': np.array(np.ma.masked_array(list_elevation).filled(np.nan))})

        grid1 = Topo(df=df).get_as_grid(remove_nans=False)
        grid2 = Topo(df=df).get_as_grid(remove_nans=True)
        vmin = -8
        vmax = 8
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        im = axes[0].imshow(grid1, cmap='ocean_r', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axes[0], shrink=.4)
        im = axes[1].imshow(grid2, cmap='ocean_r', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axes[1], shrink=.4)
        plt.suptitle(f'file: {f}, shape: {grid2.shape}')
        # plt.show()

        df.to_csv(out_xyz_path, index=False)
        np.save(out_grid_path, grid1)
        plt.savefig(out_plot_path)

        count += 1
        print(f'{count}/{n_files} \t\t {out_file}')

# counting nans at each pixel
count = 0
counter_grid = None
master_grid = None
files = os.listdir(out_grids_dir_path)
dates = []
for i in range(len(files)):
    f = sorted(files)[i]
    date = f.replace('.npy', '')
    dates.append(get_datetime_from_ymd_string(date))

    grid = np.load(os.path.join(out_grids_dir_path, f))
    if counter_grid is None:
        counter_grid = np.zeros(grid.shape)
        master_grid = np.zeros((len(files), grid.shape[0], grid.shape[1]))
    mask = np.isnan(grid)
    masked = grid * mask
    masked[np.isnan(masked)] = 1
    counter_grid += masked

    master_grid[i] = grid
    count += 1
    print(f'{count}/{n_files} \t\t {f}')

# dates = dates[1:]

ys = master_grid.shape[1]
xs = master_grid.shape[2]

target_start = None
target_end = None
interpolated_master_grid = None
for x in range(xs):
    for y in range(ys):
        pixel_series = master_grid[:, y, x]
        nan_indices = np.argwhere(np.isnan(pixel_series))#[:, 0]

        vs = np.delete(pixel_series, nan_indices)
        ds = np.delete(dates, nan_indices)

        gaps_in_days = np.diff(ds) / np.timedelta64(1, 'D')
        gaps_mask = gaps_in_days < 45
        dates_and_gaps = []

        # ranges = []
        # for i in range(len(gaps_in_days)):
        #     d1 = ds[i]
        #     d2 = ds[i+1]
        #     gap = gaps_in_days[i]
        #     keep = gaps_mask[i]
        #
        #     dates_and_gaps.append((d1, d2, gap, keep))
        #
        #     if keep:
        #         if len(ranges) == 0:
        #             ranges.append([])
        #             ranges[-1].append(d1)
        #             ranges[-1].append(d2)
        #         else:
        #             if len(ranges[-1]) == 0:
        #                 ranges[-1].append(d1)
        #                 ranges[-1].append(d2)
        #             else:
        #                 ranges[-1].append(d2)
        #     else:
        #         ranges.append([d2])
        #         ranges.append([])
        # [print(row) for row in ranges]
        # [print(row) for row in dates_and_gaps]
        # exit()

        target_dates = np.arange(ds[1], ds[-1], np.timedelta64(1, 'M'),
                                 dtype='datetime64[M]')  # Y-M only
        target_dates = target_dates.astype('datetime64[D]')  # Y-M-D
        target_dates = target_dates.astype('datetime64[s]')  # add seconds

        if interpolated_master_grid is None:
            interpolated_master_grid = np.zeros((len(target_dates), master_grid.shape[1], master_grid.shape[2]))
        if target_start is None and target_end is None:
            target_start = target_dates[0]
            target_end = target_dates[-1]
        elif target_dates[0] != target_start or target_dates[-1] != target_end:
            print(f'PROBLEM IN {x, y}')
            print(target_dates[0], target_dates[-1])

        ts_dates = [datetime_to_timestamp(d) for d in ds]
        ts_t_dates = [datetime_to_timestamp(d) for d in target_dates]

        # exit()

        f = interpolate.interp1d(ts_dates, vs)
        new_pixel_series = f(ts_t_dates)

        interpolated_master_grid[:, y, x] = new_pixel_series

np.save('counter_grid', counter_grid)
np.save('master_grid', master_grid)
np.save('interpolated_master_grid', interpolated_master_grid)
np.save('topo_dates', target_dates)

exit()
# for i in range(interpolated_master_grid.shape[0]):
#     plt.figure(figsize=(5, 3))
#     plt.imshow(interpolated_master_grid[i], vmin=-8, vmax=8)
#     plt.colorbar(shrink=1)
#     plt.suptitle(target_dates[i].astype('datetime64[D]'))
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(f'/home/mn/PycharmProjects/morphcast/init_scripts/interpolated_plots/{i}')

# TODO split raw data before interpolating according to:
#    - number of consecutive values missing
#    - storms
