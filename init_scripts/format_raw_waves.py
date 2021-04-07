import os
import json
import netCDF4
import pandas as pd


data_dir = '/media/mn/WD4TB/topo/waves/waverider-17m/data/'
output_dir = '/media/mn/WD4TB/topo/waves/waverider-17m/processed_data/'
files = os.listdir(data_dir)

n_files = len(files)
count = 1
for f in files:
    print(f)
    ncd = netCDF4.Dataset(os.path.join(data_dir, f))

    time_start = ncd.__dict__['time_coverage_start']
    time_end = ncd.__dict__['time_coverage_end']
    lat_min = ncd.__dict__['geospatial_lat_min']
    lat_max = ncd.__dict__['geospatial_lat_max']
    lng_min = ncd.__dict__['geospatial_lon_min']
    lng_max = ncd.__dict__['geospatial_lon_max']
    meta_data = {
        'time_start': time_start,
        'time_end': time_end,
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lng_min': lng_min,
        'lng_max': lng_max,
    }

    out_hs = ncd.variables['waveHs'][:]
    out_tp = ncd.variables['waveTp'][:]
    out_direction = ncd.variables['waveMeanDirection'][:]
    out_t = ncd.variables['time'][:]

    out_file_path = os.path.join(output_dir, f.replace('FRF-ocean_waves_waverider-17m_', '').replace('.nc', ''))
    df = pd.DataFrame({'t': out_t, 'hs': out_hs, 'tp': out_tp, 'dir': out_direction})

    df.to_csv(f'{out_file_path}.csv', index=False)
    with open(f'{out_file_path}.json', "w") as json_file:
        json.dump(meta_data, json_file, indent=4, sort_keys=True)

    print(f'{count}/{n_files}')
    count += 1
