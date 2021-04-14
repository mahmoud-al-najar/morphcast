import copy
import warnings
import numpy as np
import pandas as pd
from numpy.linalg import norm


class Topo:
    def __init__(self, filepath=None, df=None):
        """specify either filepath of df"""
        if filepath is not None and df is not None:
            warnings.warn('Both filepath and df are not None. Will use df only.')
            self.df = df
        elif filepath is not None:
            self.filepath = filepath
            self.df = pd.read_csv(filepath)
        elif df is not None:
            self.df = df
        else:
            warnings.warn('Specify either filepath or df. Exiting.')
            exit()
        self._alongshore_line_dfs = None
        self._average_profile_df = None
        self._grid = None

    def _get_alongshore_line_dfs(self, remove_nans=False):
        if self._alongshore_line_dfs is None:
            df = copy.deepcopy(self.df)
            if remove_nans:
                df = df[df.z.notna()]
            line_dfs = []
            while len(df) > 0:
                left_row = df.loc[df.lng.idxmin()]
                bottom_row = df.loc[df.lat.idxmin()]

                P1 = np.array([float(left_row.lng), float(left_row.lat)])
                P2 = np.array([float(bottom_row.lng), float(bottom_row.lat)])
                points = np.array([df.lng, df.lat])

                dists = []
                for index in range(points.shape[1]):
                    p = points[:, index]
                    d = (np.abs(np.cross(P2 - P1, P1 - p)) / norm(P2 - P1)) * 100
                    dists.append(d)

                df['dist'] = np.array(dists)
                line_df = df[df.dist <= 0.01]
                line_dfs.append(line_df)
                df.drop(line_df.index, inplace=True)
            self._alongshore_line_dfs = line_dfs
        return self._alongshore_line_dfs

    def get_average_cross_shore_profile(self):
        if self._average_profile_df is None:
            avg_lng = []
            avg_lat = []
            avg_z = []
            for line_df in self._get_alongshore_line_dfs():
                new_lng = np.average(line_df.lng)
                new_lat = np.average(line_df.lat)
                new_z = np.average(line_df.z)

                avg_lng.append(new_lng)
                avg_lat.append(new_lat)
                avg_z.append(new_z)
            self._average_profile_df = pd.DataFrame({'lng': avg_lng, 'lat': avg_lat, 'z': avg_z})
        return self._average_profile_df

    def get_as_grid(self, remove_nans=False):
        if self._grid is None:
            grid = []
            for line_df in self._get_alongshore_line_dfs(remove_nans):
                grid.append(np.array(line_df.z))
            grid = np.array(grid)
            grid = np.rot90(grid)
            self._grid = grid
        return self._grid
