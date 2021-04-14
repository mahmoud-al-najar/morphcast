import os
import copy
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utilities.wrappers import Topo


data_dir = '/media/mn/WD4TB/topo/survey_dems/data'
out_dir = '/media/mn/WD4TB/topo/survey_dems/xyz_data/grids'
files = os.listdir(data_dir)
n_files = len(files)
count = 0

for f in files:
    out_file = f.replace('.nc', '').replace('FRF_geomorphology_DEMs_surveyDEM_', '')
    out_file_path = os.path.join(out_dir, out_file)

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

    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # im = axes[0].imshow(grid1, cmap='ocean_r', vmin=vmin, vmax=vmax)
    # plt.colorbar(im, ax=axes[0], shrink=.4)
    # im = axes[1].imshow(grid2, cmap='ocean_r', vmin=vmin, vmax=vmax)
    # plt.colorbar(im, ax=axes[1], shrink=.4)
    # plt.suptitle(f'file: {f}, shape: {grid2.shape}')
    # plt.show()
    # plt.savefig(out_file_path)

    # df.to_csv(out_file_path + '.xyz', index=False)
    # np.save(out_file_path, grid2)

    # TODO: keep all dfs with nans, interpolate, extract subareas

    count += 1
    print(f'{count}/{n_files} \t\t {out_file_path}')
