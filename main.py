import math as m
from datetime import datetime
from itertools import chain, combinations
import read_geolife
import pandas as pd
import numpy as np
import pickle
from matplotlib import cm
import plotly.express as px

'''
Source: https://heremaps.github.io/pptk/tutorials/viewer/geolife.html
'''

label_names = {1: 'walk', 2: 'bike', 3: 'bus', 4: 'car', 5: 'subway', 6: 'train', 7: 'airplane', 8: 'boat', 9: 'run',
               10: 'motorcycle', 11: 'taxi'}

# time colormap
cols = cm.get_cmap('plasma', 48).colors
colormap = np.concatenate((cols, cols[::-1]))

# select a visualization
modes = {1: 'alt', 2: 'label', 3: 'time', 4: 'groups', 5:'simple_overlap_removal'}
mode = 6

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


# Computes distance in meters of to lon/lat points
def compute_dist(lat1, lon1, lat2, lon2):
    R = 6378.137
    dLat = lat2 * m.pi / 180 - lat1 * m.pi / 180
    dLon = lon2 * m.pi / 180 - lon1 * m.pi / 180
    a = m.sin(dLat / 2) * m.sin(dLat / 2) + \
        m.cos(lat1 * m.pi / 180) * m.cos(lat2 * m.pi / 180) * \
        m.sin(dLon / 2) * m.sin(dLon / 2)
    c = 2 * m.atan2(m.sqrt(a), m.sqrt(1 - a))
    d = R * c
    return d * 1000


def group_to_resource(g):
    name = ''
    for digit in g:
        name += str(digit)
        name += '_'
    name = name[:-1]
    return name


''' ACTUAL IMPLEMENTATIONS '''
if mode not in modes:
    df = pd.read_pickle('Data/groups_without_simple_overlap.pkl')

    df = df.sort_values(['Start', 'Finish'])
    print(df)
    groups = df['Groups'].tolist()
    start = df['Start'].tolist()
    finish = df['Finish'].tolist()
    resource = df['Resources'].tolist()
    i = 0
    counter = 0
    length = len(groups)
    while i < length - 1:
        if i == 167:
            test = 0
        j = i
        original_finish = finish[i]
        overlap_handled_one = False
        overlap_handled_two = False
        while original_finish >= start[j+1]:
            if groups[j+1].issubset(groups[i]):
                finish.append(finish[i])
                start.append(start[i])
                resource.append(resource[i])
                groups.append(groups[j+1])
                if not overlap_handled_two:
                    start[j+1] = finish[i]
                    overlap_handled_two = True

            if groups[i].issubset(groups[j+1]):
                if overlap_handled_one:
                    test = 0
                    finish.append(start[j + 1])
                    start.append(finish[j])
                    resource.append(resource[i])
                    groups.append(groups[i])
                finish.append(finish[j+1])
                start.append(start[j+1])
                resource.append(resource[j+1])
                groups.append(groups[i])
                if not overlap_handled_one:
                    finish[i] = start[j+1]
                    overlap_handled_one = True

            j += 1
        i += 1
    # print(counter)
    print(groups[245:255])
    print(start[245:255])
    print(finish[245:255])

    lists_combined = [start, finish, groups, resource]
    final_df = pd.DataFrame(lists_combined).transpose()
    final_df.columns = ['Start', 'Finish', 'Groups', 'Resources']
    final_df = final_df.sort_values(['Start', 'Finish'])
    print(final_df[245:256])
    groups = final_df['Groups'].tolist()
    group_strings = []
    for group in groups:
        string = 'Group_('
        for value in group:
            string += str(value)
            string += '-'
        string = string[:-1]
        string += ')'
        group_strings.append(string)
    final_df['Group_strings'] = group_strings
    #df_slice = final_df[245:256]
    print(start[10])
    df_slice = final_df.loc[(final_df['Start'] < '2009-04-11 23:59:00') & (final_df['Start'] > '2009-04-11 00:01:00')]
    fig = px.timeline(df_slice, x_start="Start", x_end="Finish", y="Group_strings", color="Resources")
    fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
    fig.show()
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
    df = load_df_50()
    df.sort_values(by="time", ascending=True)

    # PARAMETERS
    eps = 10  # defines how close individuals should be to be considered a group
    num = 2  # defines how many individuals forms a group
    dur = 5  # defines how long individuals should be together be considered a group
    limit = dur * 6  # defines the length of the period in which the first dur timestamps must take place

    user_list = df['user'].drop_duplicates().tolist()
    timestamps = df['time'].drop_duplicates().tolist()
    # timestamps = timestamps[3000:6000]

    # time_end, time_start = timestamps.max(), timestamps.min()
    # timestamps = pd.date_range(start=time_start, end=time_end, freq='T')
    # timestamps = timestamps[18000:2*18000] # some downscaling
    # print(timestamps)
    # print(timestamps[0])
    groups = {}
    dur_groups = {}
    filtered_groups = {}
    groups_tuples = []
    print("Done loading data\nRunning Algorithm..")

    ''' 
    FIRST LOOP COMPUTES GROUPS EACH TIMESTAMP 
    PARAMS: eps, num
    '''
    try:
        with open('data/groups.pkl', 'rb') as f:
            print("loading groups")
            groups = pickle.load(f)
    except (OSError, IOError) as e:
        for i in range(len(timestamps)):
            # Dataframe filtered to current timestamp
            dft = df.loc[df['time'] == timestamps[i]]
            users = dft['user']
            grpt = []

            if (i + 1) % 300 == 0 or i == 0:
                print(f'currently on {i + 1} of {len(timestamps)} timestamps ({timestamps[i]})')

            # Check if u1 can be grouped with u2
            for u1 in users:
                for u2 in users:
                    if u1 < u2:
                        dfu1 = dft.loc[dft['user'] == u1]
                        dfu2 = dft.loc[dft['user'] == u2]
                        lat1, lon1 = dfu1['lat'].iloc[0], dfu1['lon'].iloc[0]
                        lat2, lon2 = dfu2['lat'].iloc[0], dfu2['lon'].iloc[0]
                        r = compute_dist(lat1, lon1, lat2, lon2)
                        # print(f'u1: {u1}, (x1, y1): ({x1}, {y1})\nu2: {u2}, (x2, y2): ({x2}, {y2})')
                        # print(f'r: {r}')
                        if r <= eps:
                            added = False
                            grp1 = None
                            grp2 = None
                            for grp in grpt:
                                if u1 in grp:
                                    grp1 = grp
                                    added = True
                                if u2 in grp:
                                    grp2 = grp
                                    added = True
                            if not added:
                                grpt.append({u1, u2})
                            elif grp1 is None:
                                grp2.add(u1)
                            elif grp2 is None:
                                grp1.add(u2)
                            elif grp1 is not grp2:
                                grpt.remove(grp1)
                                grpt.remove(grp2)
                                grpt.append(grp1.union(grp2))
            # if grpt:
            #     print(f'\tgroups found: {grpt}')
            groups[timestamps[i]] = grpt

            # Filter out groups of size < num
            to_remove = []
            for grp in grpt:
                if len(grp) < num:
                    to_remove.append(grp)
            for tr in to_remove:
                # print(f"removed group {tr}")
                grpt.remove(tr)

        with open('data/groups.pkl', 'wb') as f:
            print("saving groups")
            pickle.dump(groups, f)

    # print(f'\ngroups:')
    # for i in groups:
    #     if groups[i]:
    #         print(f'{i}, {groups[i]}')

    ''' 
    SECOND LOOP COMPUTES PERSISTENT GROUPS
    PARAMS: dur, num 
    '''
    for i in range(len(timestamps) - dur):
        stamp = timestamps[i]
        grpt = groups[stamp]
        i_list = [grpt]
        for j in range(dur - 1):
            stampj = timestamps[i + j + 1]
            grps = groups[stampj]
            new_list = []

            # If stampj differs more than 2*dur minutes from stamp, disregard all groups
            diff = stampj - stamp
            diff_in_s = diff.total_seconds()
            diff_minutes = divmod(diff_in_s, 60)[0]
            if diff_minutes > limit:
                # print(f"too long: {stampj} - {stamp} = {diff_minutes}")
                break

            for grp in i_list[j]:
                for grs in grps:
                    inter = grp.intersection(grs)
                    if len(inter) >= num:
                        new_list.append(inter)
            i_list.append(new_list)

        if len(i_list) == dur:
            lst = i_list[dur - 1]
            dur_groups[stamp] = lst
        else:
            dur_groups[stamp] = []

    # filter groups
    for i in range(len(timestamps) - dur):
        stamp1 = timestamps[i]
        val1 = dur_groups[stamp1]
        if i == 0:
            filtered_groups[stamp1] = val1
        else:
            stamp2 = timestamps[i - 1]
            val2 = dur_groups[stamp2]
            if val1 != val2:
                # filtered_groups[stamp2] = val2
                filtered_groups[stamp1] = val1

    # print('\nfiltered_groups:')
    # for i in filtered_groups:
    #     print(f'{i}, {filtered_groups[i]}')
    #     if filtered_groups[i]:
    #         print(f'{i}, {filtered_groups[i]}')

    # construct tuples
    i = 0
    while i < len(timestamps) - 1 - dur:
        stamp1 = timestamps[i]
        val1 = dur_groups[stamp1]
        val2 = val1

        while val1 == val2 and i < len(timestamps) - 1 - dur:
            i += 1
            stamp2 = timestamps[i]
            val2 = dur_groups[stamp2]
            # print(f'{stamp1}: {val1}', end=" ")
            # print(f'{stamp2}: {val2}')

        # i - 2 + dur
        if val1:
            tup = (stamp1, timestamps[i - 2 + dur], val1)
            groups_tuples.append(tup)

    print('\ngroups_tuples:')
    for i in groups_tuples:
        print(f'{i}')
    df = pd.DataFrame(groups_tuples, columns=['Start', 'Finish', 'Groups'])
    print(df)
    df.to_pickle('data/groups_tuples.pkl')
elif modes[mode] == 'simple_overlap_removal':
    df = pd.read_pickle('Data/groups_tuples.pkl')
    double_groups = df[df.Groups.map(len) > 1]
    df_without_double_groups = df.drop(double_groups.index)
    groups = double_groups['Groups'].tolist()
    start = double_groups['Start'].tolist()
    finish = double_groups['Finish'].tolist()

    split_groups = []
    split_start = []
    split_finish = []

    for i in range(len(groups)):
        for g in groups[i]:
            split_start.append(start[i])
            split_finish.append(finish[i])
            split_groups.append(g)

    split_lists_combined = [split_start, split_finish, split_groups]
    split_dataframe = pd.DataFrame(split_lists_combined).transpose()
    split_dataframe.columns = ['Start', 'Finish', 'Groups']

    df_without_double_groups['Groups'] = df_without_double_groups['Groups'].apply(lambda x: x[0])
    combined_df = pd.concat([split_dataframe, df_without_double_groups])
    set_list = combined_df['Groups'].tolist()
    list_of_group_sets = []
    for s in set_list:
        if s not in list_of_group_sets:
            list_of_group_sets.append(s)

    final_df = pd.DataFrame()
    test_df = combined_df.loc[combined_df['Groups'] == {139, 108}]
    test_df = test_df.sort_values(['Start', 'Finish'])
    print(test_df)
    for group_set in list_of_group_sets:
        selection_df = combined_df.loc[combined_df['Groups'] == group_set]
        selection_df = selection_df.sort_values(['Start', 'Finish'])
        groups = selection_df['Groups'].tolist()
        start = selection_df['Start'].tolist()
        finish = selection_df['Finish'].tolist()
        new_groups = []
        new_start = []
        new_finish = []
        resource = []
        i = 0
        while i < len(groups) - 1:

            if finish[i] < start[i + 1]:
                new_start.append(start[i])
                new_finish.append(finish[i])
                new_groups.append(groups[i])
                resource.append(group_to_resource(groups[i]))
                i += 1
                if i == len(groups) - 1:
                    new_start.append(start[i])
                    new_finish.append(finish[i])
                    new_groups.append(groups[i])
                    resource.append(group_to_resource(groups[i]))
            else:
                j = i
                new_start.append(start[j])
                new_groups.append(groups[j])
                resource.append(group_to_resource(groups[j]))
                if groups[j] == {139, 108}:
                    test = 0
                assigned_finish = finish[i]

                while assigned_finish >= start[j + 1]:
                    if finish[j + 1] >= assigned_finish:
                        assigned_finish = finish[j + 1]
                    j += 1
                    if j == len(groups) - 1:
                        break
                new_finish.append(assigned_finish)
                j += 1
                i = j

        lists_combined = [new_start, new_finish, new_groups, resource]
        temp_df = pd.DataFrame(lists_combined).transpose()
        temp_df.columns = ['Start', 'Finish', 'Groups', 'Resources']
        final_df = pd.concat([final_df, temp_df])

    final_df.to_pickle('data/groups_without_simple_overlap.pkl')