# -*- coding: utf-8 -*-

##########################################################################################################################
# Pulse - random survival forests
##########################################################################################################################

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Import libraries
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Remove correlated features
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# import data, remove empty features
df = pd.read_csv( 'input/pulse_durations.csv' )
df = df.loc[:, (df != 0).any(axis=0)]

# plot correlation matrix
CM = df.corr()
mask = np.triu( np.ones_like( CM, dtype=bool ) )
fig, ax = plt.subplots( figsize=(14, 12) )
sns.set(font_scale=1.1)
ax = sns.heatmap( CM, mask=mask, annot=True, cmap='RdBu', vmin=-1, vmax=1, fmt='.2f', square=True, cbar_kws={'label': 'r value'},annot_kws={"size": 10} )
ax.figure.axes[-1].yaxis.label.set_size(20)
ax.set_facecolor('w')
plt.show()

# remove highly correlated (r >= 0.7) features
remove = [ 'rift', 'intraplate', 'ctcrust1', 'meanslope', 'shield']
df.drop( columns=remove, inplace=True )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Feature selection
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Fit data
cph = CoxPHFitter()
cph.fit(df, 'duration',event_col='end')

# Summarize results
cph.print_summary()

# Remove features with high p values (>=0.1)
remove = [ 'lava_cone', 'eruptionssince1960', 'avgrepose', 'mafic' ]
df.drop( columns=remove, inplace=True )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Train final model
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Fit 
cph.fit(df, 'duration',event_col='end')

# Summarize results
cph.print_summary()