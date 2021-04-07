import copy
import numpy as np
import pandas as pd
from numpy.linalg import norm


class Topo:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self._average_profile_df = None

    def get_average_profile(self):
        if self._average_profile_df is not None:
            return self._average_profile_df
        else:
            df = copy.deepcopy(self.df)
            df = df[df.z.notna()]

            avg_lng = []
            avg_lat = []
            avg_z = []
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
                new_lng = np.average(line_df.lng)
                new_lat = np.average(line_df.lat)
                new_z = np.average(line_df.z)

                avg_lng.append(new_lng)
                avg_lat.append(new_lat)
                avg_z.append(new_z)

                df.drop(line_df.index, inplace=True)

            self._average_profile_df = pd.DataFrame({'lng': avg_lng, 'lat': avg_lat, 'z': avg_z})
            return self._average_profile_df
