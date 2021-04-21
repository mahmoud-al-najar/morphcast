import numpy as np
import pandas as pd

def make_sub_areas(master_grid, dim, pairs=False):
    ys = master_grid.shape[1] - dim
    xs = master_grid.shape[2] - dim

    sub_area_series = []
    for x in range(xs):
        for y in range(ys):
            sub_area_series.append(master_grid[:, y:y+dim, x:x+dim])

    if pairs:
        sub_area_pairs = []
        for series in sub_area_series:
            for i in range(len(series) - 1):
                sub_area_pairs.append((series[i], series[i + 1]))
        return sub_area_pairs
    else:
        return sub_area_series


def make_dataset(topo_grids, topo_dates, waves_df, dim, pairs=False):
    # metadata setup
    ys = topo_grids.shape[1] - dim
    xs = topo_grids.shape[2] - dim
    waves_df['date'] = pd.to_datetime(waves_df['date'])

    dataset = []
    for i in range(len(topo_dates) - 1):
        interval_dates = (topo_dates[i], topo_dates[i+1])
        mask = (waves_df.date > interval_dates[0]) & (waves_df.date <= interval_dates[1])
        interval_waves_df = waves_df.loc[mask]

        if len(interval_waves_df) == 24 * 7 * 4:
            # print(f'{i}/{len(topo_dates) - 1}\t{interval_dates}')
            grid1 = topo_grids[i]
            grid2 = topo_grids[i + 1]
            sub_area_series = []
            for x in range(xs):
                for y in range(ys):
                    subarea1 = grid1[y:y + dim, x:x + dim]
                    subarea2 = grid2[y:y + dim, x:x + dim]
                    dataset.append((subarea1,
                                    subarea2,
                                    interval_waves_df.hs.values,
                                    interval_waves_df.tp.values,
                                    interval_waves_df.dir.values))
    # print(len(dataset))
    return dataset
