# -*- coding: utf-8 -*-

##########################################################################################################################
# Event - XGBoost model using accelerated failure time
##########################################################################################################################

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Import libraries
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import xgboost as xgb
from xgbse import XGBSEKaplanNeighbors
from xgbse.converters import convert_data_to_xgb_format, convert_to_structured
from xgbse.metrics import concordance_index, approx_brier_score
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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
X = df.drop(['duration', 'end'], axis=1)
y = convert_to_structured(df['duration'], df['end'])

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Visualize model parameters
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
seed = 0

# make train/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# convert to xgboost format
d_train = convert_data_to_xgb_format(X_train, y_train, 'survival:aft')
d_test = convert_data_to_xgb_format(X_test, y_test, 'survival:aft')

params = {'objective': 'survival:aft',
          'eval_metric': 'aft-nloglik',
          'aft_loss_distribution': 'normal',
          'aft_loss_distribution_scale': 1.20,
          'tree_method': 'hist',
          'learning_rate': 0.05, 'max_depth': 2}

scores_ci = {}
for eta in np.logspace(-2,0,10):
    params['learning_rate'] = eta
    model = xgb.train(params, d_train, num_boost_round=5,
                evals=[(d_train, 'train')])
    pred = model.predict(d_test)
    scores_ci[eta] = concordance_index(y_test, -pred, risk_strategy='precomputed')

x_plot, y_plot = zip(*scores_ci.items())
plt.plot(x_plot, y_plot)
plt.xlabel('eta')
plt.ylabel('Concordance Index')
plt.show()

params['learning_rate'] = max(scores_ci, key=scores_ci. get)
scores_ci = {}
for aft_loss_distribution_scale in np.linspace(0.5,1.5,10):
    params['aft_loss_distribution_scale'] = aft_loss_distribution_scale
    model = xgb.train(params, d_train, num_boost_round=5,
                evals=[(d_train, 'train')])
    pred = model.predict(d_test)
    scores_ci[aft_loss_distribution_scale] = concordance_index(y_test, -pred, risk_strategy='precomputed')

x_plot, y_plot = zip(*scores_ci.items())
plt.plot(x_plot, y_plot)
plt.xlabel('aft_loss_distribution_scale')
plt.ylabel('Concordance Index')
plt.show()


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Optimization
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Initialize random number generator, once for each repeat of 5-fold cross validation
seed = 0

aft_loss_distribution = ['normal','logistic']
aft_loss_distribution_scale  = np.linspace(0.8,1.3,5)
learning_rate  = np.logspace(-0.5,0,5)
max_depth = np.arange(1,4)
n = len(aft_loss_distribution)*len(aft_loss_distribution_scale)*len(learning_rate)*len(max_depth)


kf = KFold(5, shuffle=True, random_state=seed)
kf.get_n_splits(y)
count = 1
score = 0


# convert to xgboost format
d_train = convert_data_to_xgb_format(X_train, y_train, 'survival:aft')
d_test = convert_data_to_xgb_format(X_test, y_test, 'survival:aft')

for ald in aft_loss_distribution:
    for alds in aft_loss_distribution_scale:
        for lr in learning_rate:
            for md in max_depth:
                print( '{0} of {1}'.format(count,n) )
                params['aft_loss_distribution'] = ald
                params['aft_loss_distribution_scale'] = alds
                params['learning_rate'] = lr
                params['max_depth'] = md
                scores = []
                for train_index, test_index in kf.split(y):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    d_train = convert_data_to_xgb_format(X_train, y_train, 'survival:aft')
                    d_test = convert_data_to_xgb_format(X_test, y_test, 'survival:aft')
                    model = xgb.train(params, d_train, num_boost_round=5,
                                evals=[(d_train, 'train')], verbose_eval=False)
                    pred = model.predict(d_test)
                    s = concordance_index(y_test, -pred, risk_strategy='precomputed')
                    scores.append(s)
                if np.mean(scores) > score:
                    score = np.mean(scores)
                    best_ald = ald
                    best_alds = alds
                    best_lr = lr
                    best_md = md
                count += 1
                print(np.mean(scores))


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Cross validation, concordance index and brier score
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Initialize random number generator, once for each repeat of 5-fold cross validation
random_states = [ 20,21,22,23,24 ]

# Cross validation
results_c = [] #concordance index
results_b = [] #brier score

params['aft_loss_distribution'] = best_ald
params['aft_loss_distribution_scale'] = best_alds
params['learning_rate'] = best_lr
params['max_depth'] = best_md

for seed in random_states:

    kf = KFold(5, shuffle=True, random_state=seed)
    kf.get_n_splits(y)

    for train_index, test_index in kf.split(y):
        print( '{0} of {1}'.format(len(results_c)+1, 5*len(random_states)) )

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        d_train = convert_data_to_xgb_format(X_train, y_train, 'survival:aft')
        d_test = convert_data_to_xgb_format(X_test, y_test, 'survival:aft')

        model = xgb.train(params, d_train, num_boost_round=5,
                    evals=[(d_train, 'train')], verbose_eval=False)

        pred = model.predict(d_test)
        ci = concordance_index(y_test, -pred, risk_strategy='precomputed')
        results_c.append(ci)

# Print results
print( 'Average concordance index ({0} repeats of 5-fold cross validation): {1}'.format( len(random_states), round(np.mean(results_c),4) ) )
print( 'Standard deviation: {}'.format( round(np.std(results_c,ddof=1),4) ) )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Train final model
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Organize data
d = convert_data_to_xgb_format(X, y, 'survival:aft')

# Train model
model = xgb.train(params, d, num_boost_round=5,
                  evals=[(d, 'train')], verbose_eval=False)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Calculate shap values
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
explainer = shap.Explainer(model)
shaps = explainer(X)
shap.summary_plot(shaps, X)
