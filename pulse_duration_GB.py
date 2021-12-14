# -*- coding: utf-8 -*-

##########################################################################################################################
# Pulse - Gradient boosted models using Cox's partial likelihood
##########################################################################################################################

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Import libraries
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Test for correlated features
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

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Prepare data
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# remove highly correlated (r >= ~0.7) features
remove = [ 'rift', 'intraplate', 'ctcrust1', 'meanslope', 'shield']
df.drop( columns=remove, inplace=True )

# convert duration from days to seconds
df.duration *= 24*60*60

# Prepare variables
d = df.loc[:,['end', 'duration']]
d.end = d.end == 1
y = d.to_records(index=False)
Xt = df.copy()
Xt.drop( columns=['duration','end'], inplace=True )
feature_names = Xt.columns.tolist()

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Visualize n_estimators
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
seed = 1

X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, random_state=seed)

scores_cph_tree = {}

model = GradientBoostingSurvivalAnalysis(
                                            learning_rate=1.0, max_depth=1, random_state=0
                                        )
for n_estimators in range(5, 205, 5):
    model.set_params(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    scores_cph_tree[n_estimators] = model.score(X_test, y_test)

x, y = zip(*scores_cph_tree.items())
plt.plot(x, y)
plt.xlabel('n_estimator')
plt.ylabel('Concordance Index')
plt.show()

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Optimization
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Need to do

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Cross validation, concordance index and brier score
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Initialize random number generator, once for each repeat of 5-fold cross validation
random_states = [ 20,21,22,23,24 ]
Xt = Xt.values

# Cross validation
results_c = [] #concordance index
results_b = [] #brier score

for seed in random_states:
    
    model = GradientBoostingSurvivalAnalysis(
                                                learning_rate=1.0, max_depth=1, random_state=seed+1, n_estimators=100
                                            )

    kf = KFold(5, shuffle=True, random_state=seed)
    kf.get_n_splits(y)
    
    for train_index, test_index in kf.split(y):
        print( '{0} of {1}'.format(len(results_c)+1, 5*len(random_states)) )
        
        X_train, X_test = Xt[train_index], Xt[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        model.fit(X_train, y_train)
        
        results_c.append(model.score(X_test,y_test))
        
        #filter test dataset so we only consider event times within the range given by the training datasets for brier score
        mask = (y_test.field(1) >= min(y_train.field(1))) & (y_test.field(1) <= max(y_train.field(1)))
        X_test = X_test[mask]
        y_test = y_test[mask]
        
        survs = model.predict_survival_function(X_test)
        times = np.linspace( min([time[1] for time in y_test]), max([time[1] for time in y_test])*.999, 100 )
        preds = np.asarray( [ [sf(t) for t in times] for sf in survs ] )
        score = integrated_brier_score(y_train, y_test, preds, times)
        
        results_b.append( score )

# Print results
print( 'Average concordance index ({0} repeats of 5-fold cross validation): {1}'.format( len(random_states), round(np.mean(results_c),4) ) )
print( 'Standard deviation: {}'.format( round(np.std(results_c,ddof=1),4) ) )
print( 'Average Brier score for ({0} repeats of 5-fold cross validation): {1}'.format( len(random_states), round(np.mean(results_b),4) ) )
print( 'Standard deviation: {}'.format( round(np.std(results_b,ddof=1),4) ) )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Train final model
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Set up model
model = GradientBoostingSurvivalAnalysis(
                                            learning_rate=1.0, max_depth=1, random_state=0, n_estimators=100
                                        )

# Train model on entire dataset
model.fit(Xt, y)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Calculate shap values
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

explainer = shap.Explainer(model.predict, Xt, feature_names=feature_names)
shaps = explainer(Xt)
shap.summary_plot(shaps, Xt)