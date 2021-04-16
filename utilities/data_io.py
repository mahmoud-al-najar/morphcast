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
