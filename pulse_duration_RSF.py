# -*- coding: utf-8 -*-

##########################################################################################################################
# Pulse - Random Survival Forests
##########################################################################################################################

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Import libraries
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import integrated_brier_score
from sklearn.model_selection import KFold, GridSearchCV
from eli5.sklearn import PermutationImportance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Remove correlated features and prepare data
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# import data, remove empty features
df = pd.read_csv( 'input/pulse_durations.csv' )
df = df.loc[:, (df != 0).any(axis=0)]

# convert duration from days to seconds
df.duration *= 24*60*60

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

# Prepare data
d = df.loc[:,['end', 'duration']]
d.end = d.end == 1
y = d.to_records(index=False)
Xt = df.copy()
Xt.drop( columns=['duration','end'], inplace=True )
feature_names = Xt.columns.tolist()
Xt = Xt.values

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Initial optimization
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Initialize random number generator
seed = 0

# Make a small grid
min_samples_split = [5,10,20]
min_samples_leaf = [2,3,4,5]
param_grid = dict( min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf )

# Grid search
grid = GridSearchCV(estimator=RandomSurvivalForest( n_estimators=1000,
                                                    max_features='sqrt',
                                                    n_jobs=-1,
                                                    random_state=1 ),
                    param_grid=param_grid,
                    cv=KFold(random_state=seed, shuffle=True),
                    verbose=10)
grid_results = grid.fit( Xt, y )

# Assess results
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
best = grid_results.best_params_

for mean, stdev, param in zip(means, stds, params):
    print( '{0} ({1}) with: {2}'.format(round(mean,3), round(stdev,3), param) )

print( 'Best: {0}, using {1}'.format(grid_results.best_score_, best) )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Feature selection
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Initialize random number generator
seed = 10

# Create a list that holds results of feature selection
results = []

# Set up 5-fold cross validation
kf = KFold(n_splits=5,shuffle=True,random_state=seed)
kf.get_n_splits(y)

# Loop through each fold
count = 0
for train_index, test_index in kf.split(y):
    count += 1
    print(  '{0} of {1}'.format(count, 5) )
    X_train, X_test = Xt[train_index], Xt[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # This is the random forest model
    rsf = RandomSurvivalForest( n_estimators=1000,
                                min_samples_split=best['min_samples_split'],
                                min_samples_leaf=best['min_samples_leaf'],
                                max_features='sqrt',
                                n_jobs=-1,
                                random_state=seed+1)
    rsf.fit(X_train, y_train)

    # Use permutation importance to assess features
    perm = PermutationImportance(rsf, n_iter=15, random_state=seed+2)
    perm.fit(X_test, y_test)

    # Save results
    result = [ [x[0], x[1], x[2]] for x in zip(perm.feature_importances_, perm.feature_importances_std_, feature_names) ]
    result.sort(key=lambda x: x[0], reverse=True)
    results.append(result)

# Organize results
perm = [[],[],[]]
labels = []
avg = []
for feature in feature_names:
    perm[0].append( feature )
    perm[1].append( [] )
    perm[2].append( [] )
    for result in results:
        rT = list(map(list, zip(*result)))
        perm[1][-1].append( rT[0][ rT[2].index(feature) ] )
        perm[2][-1].append( rT[1][ rT[2].index(feature) ] )
perm[1] = [ np.mean(x) for x in perm[1] ]
for i in range(len(perm[1])):
    m = max( perm[1] )
    avg.append( m )
    labels.append( perm[0][ perm[1].index(m) ] )
    perm[0].pop( perm[1].index(m) )
    perm[1].pop( perm[1].index(m) )

# Plot results
plt.scatter( range( len( avg ) ), avg, zorder=10 )
plt.xticks( range(len( avg ) ), labels, rotation = 60, ha='right', rotation_mode="anchor" )
plt.axhline( 0, color='k', linestyle='--', lw=0.8, zorder=0 )
plt.show()

# Remove features that negatively affect the model
remove = [ 'complex', 'felsic', 'eruptionssince1960', 'stratovolcano', 'volume' ]
df.drop( columns=remove, inplace=True )


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Final optimization
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Initialize random number generator
seed = 0

# Make a grid
min_samples_split = [5,10,15,20]
min_samples_leaf = [4,5,6]
n_estimators = [750,1000,1250]
param_grid = dict( min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators )

# Grid search
grid = GridSearchCV(estimator=RandomSurvivalForest( max_features='sqrt',
                                                    n_jobs=-1,
                                                    random_state=1 ),
                    param_grid=param_grid,
                    cv=KFold(random_state=seed, shuffle=True),
                    verbose=10)
grid_results = grid.fit( Xt, y )

# Assess results
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
best = grid_results.best_params_

for mean, stdev, param in zip(means, stds, params):
    print( '{0} ({1}) with: {2}'.format(round(mean,3), round(stdev,3), param) )

print( 'Best: {0}, using {1}'.format(grid_results.best_score_, best) )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Cross validation, concordance index and brier score
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Prepare data
d = df.loc[:,['end', 'duration']]
d.end = d.end == 1
y = d.to_records(index=False)
Xt = df.copy()
Xt.drop( columns=['duration','end'], inplace=True )
feature_names = Xt.columns.tolist()
Xt = Xt.values

# Initialize random number generator, once for each repeat of 5-fold cross validation
random_states = [ 20,21,22,23,24 ]

# Cross validation
results_c = [] #concordance index
results_b = [] #brier score
for seed in random_states:

    model = RandomSurvivalForest(n_estimators=best['n_estimators'],
                                 min_samples_split=best['min_samples_split'],
                                 min_samples_leaf=best['min_samples_leaf'],
                                 max_features='sqrt',
                                 n_jobs=-1,
                                 random_state=seed+1)

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
        times = np.linspace( np.percentile([time[1] for time in y_test], 25), np.percentile([time[1] for time in y_test], 75), 10 )
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
rsf = RandomSurvivalForest(n_estimators=1000,
                            min_samples_split=best['min_samples_split'],
                            min_samples_leaf=best['min_samples_leaf'],
                            max_features='sqrt',
                            n_jobs=-1)
# Train model on entire dataset
rsf.fit(Xt, y)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Calculate shap values
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

explainer = shap.Explainer(rsf.predict, Xt, feature_names=feature_names)
shaps = explainer(Xt)
shap.summary_plot(shaps, Xt)
