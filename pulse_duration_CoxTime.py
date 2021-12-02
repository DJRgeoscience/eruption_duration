# -*- coding: utf-8 -*-

##########################################################################################################################
# Pulse - Cox Proportional Harazards Model with Ridge Penalty
##########################################################################################################################

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Import libraries
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import KFold
import torch
import torchtuples as tt
from pycox.datasets import metabric
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv
import matplotlib.pyplot as plt
import seaborn as sns


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

# Set random seeds
np.random.seed(1)
_ = torch.manual_seed(2)

# Prepare training and test datasets
df_train = df.copy()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

# Standardize
cols_standardize = [ 'elevation', 'eruptionssince1960', 'avgrepose', 'h_bw', 'ellip' ]
cols_leave = ['explosive', 'continuous', 'stratovolcano', 'subduction', 'continental', 'mafic', 'intermediate', 'felsic', 'summit_crater' ]

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

labtrans = CoxTime.label_transform()
get_target = lambda df: (df['duration'].values, df['end'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))
durations_test, events_test = get_target(df_test)
val = tt.tuplefy(x_val, y_val)

val.shapes()
val.repeat(2).cat().shapes()

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Select learning rate
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

in_features = x_train.shape[1]
num_nodes = [32, 32]
batch_norm = True
dropout = 0.1
net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)

model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
batch_size = 256
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=2)
_ = lrfinder.plot()
lrfinder.get_best_lr()

#-----------------------------------------------------------------------------
#%% View partial log-liklihood
#-----------------------------------------------------------------------------

model.optimizer.set_lr(0.01)
epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val.repeat(10).cat())
_ = log.plot()
model.partial_log_likelihood(*val).mean()

#-----------------------------------------------------------------------------
#%% Prediction and CI
#-----------------------------------------------------------------------------

_ = model.compute_baseline_hazards()
surv = model.predict_surv_df(x_test)
surv.iloc[:, :5].plot()
plt.ylabel('Exceedance probability')
_ = plt.xlabel('Time')
plt.xscale('log')
plt.show()

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
print( ev.concordance_td() )

#-----------------------------------------------------------------------------
#%% Cox-time with 5 repeats of 5-fold cross-validation
#-----------------------------------------------------------------------------

out = []
count = 1
for random_state in range(5):
    _ = torch.manual_seed(random_state+20)
    
    kf = KFold(n_splits=5,shuffle=True,random_state=random_state)
    kf.get_n_splits(df)
    
    for train_index, test_index in kf.split(df):
        print(f'{count} of 25')
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_val = x_mapper.transform(df_val).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')
        
        labtrans = CoxTime.label_transform()
        get_target = lambda df: (df['duration'].values, df['end'].values)
        y_train = labtrans.fit_transform(*get_target(df_train))
        y_val = labtrans.transform(*get_target(df_val))
        durations_test, events_test = get_target(df_test)
        val = tt.tuplefy(x_val, y_val)
        
        val.shapes()
        val.repeat(2).cat().shapes()
        
        in_features = x_train.shape[1]
        num_nodes = [32, 32]
        batch_norm = True
        dropout = 0.1
        net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
        
        model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
        batch_size = 256
        lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=2)
        lrfinder.get_best_lr()

        model.optimizer.set_lr(0.01)
        epochs = 512

        callbacks = [tt.callbacks.EarlyStopping()]
        verbose = False
        log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                        val_data=val.repeat(10).cat())
        model.partial_log_likelihood(*val).mean()
    
        _ = model.compute_baseline_hazards()
        surv = model.predict_surv_df(x_test)

        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        out.append( ev.concordance_td() )
        count += 1

print( np.mean(out) )
print( np.std(out,ddof=1) )

