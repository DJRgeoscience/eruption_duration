# -*- coding: utf-8 -*-

##########################################################################################################################
# Event - Componentwise gradient boosted models using Cox's partial likelihood
##########################################################################################################################

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Import libraries
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

from sklearn.model_selection import train_test_split
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis as cgb
from sklearn.model_selection import KFold, cross_val_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Test for correlated features
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# import data, remove empty features
df = pd.read_csv( 'input/event_durations.csv' )
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

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Prepare data
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# remove highly correlated (r >= ~0.7) features
remove = [ 'meanslope', 'rift', 'stratovolcano' ]
df.drop( columns=remove, inplace=True )

# Prepare variables
d = df.loc[:,['end', 'duration']]
d.end = d.end == 1
y = d.to_records(index=False)
Xt = df.copy()
Xt.drop( columns=['duration','end'], inplace=True )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Visualize n_estimators
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
seed = 5

X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, random_state=seed)

scores_cph_tree = {}

est_cph_tree = cgb(
                    learning_rate=1.0, random_state=0
                  )
for n_estimators in range(10, 510, 10):
    est_cph_tree.set_params(n_estimators=n_estimators)
    est_cph_tree.fit(X_train, y_train)
    scores_cph_tree[n_estimators] = est_cph_tree.score(X_test, y_test)

x, y = zip(*scores_cph_tree.items())
plt.plot(x, y)
plt.xlabel('n_estimator')
plt.ylabel('Concordance Index')
plt.show()

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Cross validation
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Initialize random number generator, once for each repeat of 5-fold cross validation
random_states = [ 20,21,22,23,24 ]

# Cross validation
results = []
for seed in random_states:
    print(  '{0} of {1}'.format(len(results)+1, len(random_states)) )
    kf = KFold(5, shuffle=True, random_state=seed)
    model = cgb(
                    learning_rate=1.0, random_state=seed, n_estimators=300
               )
    result = cross_val_score(model, Xt, y, cv=kf)
    results.append( result.tolist() )

print( 'Average concordance index ({0} repeats of 5-fold cross validation): {1}'.format( len(random_states), round(np.mean(results),2) ) )
print( 'Standard deviation: {}'.format( round(np.std(results,ddof=1),2) ) )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Train final model
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Set up model
est_cph_tree = cgb(
                    learning_rate=1.0, random_state=0, n_estimators=120
                  )

# Train model on entire dataset
est_cph_tree.fit(Xt, y)
  