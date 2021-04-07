import os
import netCDF4
import numpy as np
import pandas as pd


data_dir = '/media/mn/WD4TB/topo/survey_dems/data'
out_dir = '/media/mn/WD4TB/topo/survey_dems/xyz_data'
files = os.listdir(data_dir)
n_files = len(files)

count = 0
for f in files:
    out_file = f.replace('.nc', '.xyz').replace('FRF_geomorphology_DEMs_surveyDEM_', '')
    out_file_path = os.path.join(out_dir, out_file)

    if True:
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
        df = pd.DataFrame({'lng': np.array(list_lng), 'lat': np.array(list_lat), 'z': np.array(list_elevation)})
        df.to_csv(out_file_path, index=False)
    count += 1
    print(f'{count}/{n_files}')
