# -*- coding: utf-8 -*-

##########################################################################################################################
# Pulse - Cox Proportional Harazards Model
##########################################################################################################################

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Import libraries
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

from sksurv.linear_model import CoxPHSurvivalAnalysis as CPHSA
from sksurv.metrics import integrated_brier_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
from lifelines import CoxPHFitter as CPHF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Remove correlated features
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
remove = [ 'rift', 'intraplate', 'ctcrust1', 'meanslope', 'shield', 'complex']
df.drop( columns=remove, inplace=True )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Feature selection
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Fit data
cph = CPHF()
cph.fit(df, 'duration',event_col='end')

# Summarize results
cph.print_summary()

# Remove features with high p values (>=0.03)
remove = [ 'stratovolcano', 'dome', 'lava_cone', 'subduction', 'eruptionssince1960', 'avgrepose', 'mafic', 'intermediate', 'felsic', 'summit_crater', 'volume' ]
df.drop( columns=remove, inplace=True )

# Prepare data for cross validation
d = df.loc[:,['end', 'duration']]
d.end = d.end == 1
y = d.to_records(index=False)
Xt = df.copy()
Xt.drop( columns=['duration','end'], inplace=True )
feature_names = Xt.columns.tolist()
Xt = Xt.values

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Cross validation, concordance index
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Initialize random number generator
seed = 1

kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)

model = CPHSA()
results = cross_val_score(model, Xt, y, cv=kf)

# Print results
print( 'Average concordance index for ({0} repeats of 5-fold cross validation): {1}'.format( 5, round(results.mean(),4) ) )
print( 'Standard deviation: {}'.format( round(results.std(ddof=1),4) ) )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Cross validation, integrated brier score
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

random_states = [11,12,13,14,15]
results = []
model = CPHSA()

for seed in random_states:
    kf = KFold(n_splits=5,shuffle=True,random_state=seed)
    kf.get_n_splits(y)
    for train_index, test_index in kf.split(y):
        print( '{0} of {1}'.format(len(results)+1, 5*len(random_states)) )
        
        X_train, X_test = Xt[train_index], Xt[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        model.fit(X_train, y_train)
        
        #filter test dataset so we only consider event times within the range given by the training datasets
        mask = (y_test.field(1) >= min(y_train.field(1))) & (y_test.field(1) <= max(y_train.field(1)))
        X_test = X_test[mask]
        y_test = y_test[mask]
        
        survs = model.predict_survival_function(X_test)
        times = np.linspace( min([time[1] for time in y_test]), max([time[1] for time in y_test])*.999, 100 )
        preds = np.asarray( [ [sf(t) for t in times] for sf in survs ] )
        score = integrated_brier_score(y_train, y_test, preds, times)
        
        results.append( score )

# Print results
print( 'Average Brier score for ({0} repeats of 5-fold cross validation): {1}'.format( len(random_states), round(np.mean(results),4) ) )
print( 'Standard deviation: {}'.format( round(np.std(results,ddof=1),4) ) )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Train final model
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

model = CPHSA()
model.fit(Xt,y)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Calculate shap values
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

explainer = shap.Explainer(model.predict, Xt, feature_names=feature_names)
shaps = explainer(Xt)
shap.summary_plot(shaps, Xt)