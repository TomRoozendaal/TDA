import read_geolife
import pandas as pd
import numpy as np
import pptk

'''Load/read data'''
try:
    df = pd.read_pickle('data/geolife.pkl')
except (OSError, IOError) as e:
    df = read_geolife.read_all_users('data')
    df.to_pickle('data/geolife.pkl')

print(df)

P = np.c_[df['lon'], df['lat'], np.zeros(len(df))]
viewer = pptk.viewer(P)

'''color each point based on its altitude'''
viewer.attributes(df['alt'])

'''give it a colormap; manually set the map domain'''
viewer.color_map('jet', scale=[0, 20000])