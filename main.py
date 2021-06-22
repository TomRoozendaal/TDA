import read_geolife
import pandas as pd
import numpy as np
import pptk
import pyproj
from matplotlib import cm

'''
Source: https://heremaps.github.io/pptk/tutorials/viewer/geolife.html
'''

label_names = {1: 'walk', 2: 'bike', 3: 'bus', 4: 'car', 5: 'subway', 6: 'train', 7: 'airplane', 8: 'boat', 9: 'run',
               10: 'motorcycle', 11: 'taxi'}

# time colormap
cols = cm.get_cmap('plasma', 48).colors
colormap = np.concatenate((cols, cols[::-1]))

# select a visualization
modes = {1: 'alt', 2: 'label', 3: 'time'}
mode = 3

# Load full data frame
def load_df():
    try:
        d = pd.read_pickle('data/geolife.pkl')
    except (OSError, IOError) as e:
        d = read_geolife.read_all_users('data')
        d.to_pickle('data/geolife.pkl')
    return d

# Load data frame masked to UTM zone 50
def load_df_50():
    try:
        d50 = pd.read_pickle('data/geolife_50.pkl')
    except (OSError, IOError) as e:
        d = load_df()
        # Only consider points inside UTM zone 50
        mask_50 = (d['lon'] > 114.0) & (d['lon'] < 120.0) & (d['lat'] > 32.0) & (d['lat'] < 48.0)
        d50 = d[mask_50]
        d50.to_pickle('data/geolife_50.pkl')
    return d50

# Load data frame with coordinates lat1, lon1 and lat2, lon2 in specific regions with threshold
def load_df_route(lat1, lon1, lat2, lon2, threshold):
    try:
        d_route = pd.read_pickle('data/geolife_route.pkl')
    except (OSError, IOError) as e:
        d = load_df()

        mask_50 = (d['lon'] > 114.0) & (d['lon'] < 120.0) & (d['lat'] > 32.0) & (d['lat'] < 48.0)
        d50 = d[mask_50]

        df_route_1 = d50.query('lon > @lon1 - @threshold & lon < @lon1 + @threshold & lat > @lat1 - @threshold & lat < @lat1 + @threshold')
        df_route_2 = d50.query('lon > @lon2 - @threshold & lon < @lon2 + @threshold & lat > @lat2 - @threshold & lat < @lat2 + @threshold')
        users1 = df_route_1['user'].unique()
        users2 = df_route_2['user'].unique()
        users = np.intersect1d(users1, users2)

        route = d50[d50['user'].isin(users)]
        final_route = route.iloc[:0].copy()
        for user in route['user'].unique():
            user_datapoints = route.query('user == @user')
            time_area_1 = user_datapoints.query('(lon > @lon1 - @threshold & lon < @lon1 + @threshold & lat > @lat1 - @threshold & lat < @lat1 + @threshold)').iloc[-1].time
            time_area_2 = user_datapoints.query('(lon > @lon2 - @threshold & lon < @lon2 + @threshold & lat > @lat2 - @threshold & lat < @lat2 + @threshold)').iloc[0].time
            final_datapoints = user_datapoints.query('(time >= @time_area_1 & time <= @time_area_2) | (time >= @time_area_2 & time <= @time_area_1)')
            final_route = final_route.append(final_datapoints)

        final_route.to_pickle('data/geolife_route.pkl')
    return final_route

if mode not in modes:
    print("mode not available")
elif modes[mode] == 'alt':
    '''Load/read data'''
    df = load_df()
    print(df)
    print("Done loading data\nSetting up viewer")

    P = np.c_[df['lon'], df['lat'], np.zeros(len(df))]
    viewer = pptk.viewer(P)

    # Color each point based on its altitude
    viewer.attributes(df['alt'])

    # Give it a colormap; manually set the map domain
    viewer.color_map('jet', scale=[0, 20000])
elif modes[mode] == 'label':
    '''Load/read data'''
    df = load_df_50()
    print(df)
    print("Done loading data\nConverting data")

    # Convert lat/lon to meters in UTM coordinate system
    # Note: takes time!
    # proj = pyproj.Proj(proj='utm', zone=50, ellps='WGS84')
    # x, y = proj(df['lon'].tolist(), df['lat'].tolist())
    # P = np.c_[x, y, 0.3048 * df['alt']]  # feet are converted to meters

    P = np.c_[df['lon'], df['lat'], np.zeros(len(df))]
    print("Conversion done\nSetting up viewer")

    # Setup viewer; only show labelled data
    print(label_names)
    mask_labelled = df['label'] != 0
    viewer = pptk.viewer(P[mask_labelled])
    viewer.attributes(df[mask_labelled]['label'])
elif modes[mode] == 'time':
    '''Load/read data'''
    df = load_df_route(39.917846, 116.397032, 39.883173, 116.412876, 0.005530)
    print(df)
    print("Done loading data\nSetting up viewer")

    P = np.c_[df['lon'], df['lat'], np.zeros(len(df))]
    hours = df['time'].dt.hour + round(df['time'].dt.minute / 15) * 0.25

    # Setup viewer; only show labelled data
    viewer = pptk.viewer(P)
    viewer.attributes(hours)
    viewer.color_map(colormap)