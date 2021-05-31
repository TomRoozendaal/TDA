import math as m
from itertools import chain, combinations
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
modes = {1: 'alt', 2: 'label', 3: 'time', 4: 'groups'}
mode = 4

''' AUXILIARY FUNCTIONS '''
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


# Return powerset of a set
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    lst = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
    result = []
    for s in lst:
        if len(s) > 0:
            result.append(set(s))
    return result

''' ACTUAL IMPLEMENTATIONS '''
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
    df = load_df_50()
    print(df)
    print("Done loading data\nSetting up viewer")

    P = np.c_[df['lon'], df['lat'], np.zeros(len(df))]
    hours = df['time'].dt.hour + round(df['time'].dt.minute / 15) * 0.25

    # Setup viewer; only show labelled data
    viewer = pptk.viewer(P)
    viewer.attributes(hours)
    viewer.color_map(colormap)
elif modes[mode] == 'groups':
    '''Load/read data'''
    df = load_df()

    # PARAMETERS
    eps = 0.01  # defines how close individuals should be to be considered a group
    dur = 3  # defines how long individuals should be together be considered a group
    num = 2  # defines how many individuals forms a group

    # users = df['user'].drop_duplicates()
    time_stamps = df['time'].drop_duplicates()
    groups = {}

    # some downscaling, to be removed
    time_stamps = time_stamps.head(30)
    # print(users)
    print(time_stamps)
    print("Done loading data\nRunning Algorithm..")
    ''' 
    FIRST LOOP COMPUTES GROUPS EACH TIMESTAMP 
    PARAMS: eps 
    '''
    for i in range(len(time_stamps)):
        print(f'currently on {i + 1} of {len(time_stamps)} timestamps')
        # For a specific timestamp, check if u1 can be grouped with u2

        # print(time_stamps[i])
        # Dataframe filtered to current timestamp
        dft = df.loc[df['time'] == time_stamps[i]]
        users = dft['user']
        grpt = []
        # print(users)

        for u1 in users:
            for u2 in users:
                if u1 < u2:
                    dfu1 = dft.loc[dft['user'] == u1]
                    dfu2 = dft.loc[dft['user'] == u2]
                    # TODO: convert lat/long to meters (x/y)
                    # x1, y1 = dfu1['x'].iloc[0], dfu1['y'].iloc[0]
                    # x2, y2 = dfu2['x'].iloc[0], dfu2['y'].iloc[0]
                    x1, y1 = dfu1['lat'].iloc[0], dfu1['lon'].iloc[0]
                    x2, y2 = dfu2['lat'].iloc[0], dfu2['lon'].iloc[0]
                    r = m.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    print(f'u1: {u1}, (x1, y1): ({x1}, {y1})\nu2: {u2}, (x2, y2): ({x2}, {y2})')
                    print(f'r: {r}')
                    if r < eps:
                        added = False
                        for grp in grpt:
                            if u1 in grp or u2 in grp:
                                grp.add(u1)
                                grp.add(u2)
                                added = True
                        if not added:
                            grpt.append({u1, u2})
        groups[time_stamps[i]] = grpt

    print(groups)

    ''' 
    SECOND LOOP COMPUTES OVERALL GROUPS     
    PARAMS: dur, num 
    '''
    # TODO: finish
    for i in range(len(time_stamps) - dur):
        stamp = time_stamps[i]
        grpt = groups[stamp]
        for grp in grpt:
            print(grp)
            pset = powerset(grp)
            print(pset)

            for j in range(dur):
                stampj = time_stamps[i + j + 1]


