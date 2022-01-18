# -*- coding: utf-8 -*-

##########################################################################################################################
# Event - Survival analysis with neural networks and Cox regression (Cox-Time)
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
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv
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
cols_standardize = [ 'vei', 'repose', 'ctcrust1', 'elevation', 'volume', 'eruptionssince1960', 'avgrepose',
                   'h_bw', 'ellip']
cols_leave = [ 'caldera', 'dome', 'shield', 'complex', 'lava_cone', 'compound', 'subduction', 'intraplate',
             'continental', 'mafic', 'intermediate', 'felsic', 'summit_crater']

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

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% View partial log-liklihood
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

model.optimizer.set_lr(0.01)
epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val.repeat(10).cat())
_ = log.plot()
model.partial_log_likelihood(*val).mean()

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Prediction and CI
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

_ = model.compute_baseline_hazards()
surv = model.predict_surv_df(x_test)
surv.iloc[:, :5].plot()
plt.ylabel('Exceedance probability')
_ = plt.xlabel('Time')
plt.xscale('log')
plt.show()

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
print( ev.concordance_td() )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Optimization
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Initialize random number generator, once for each repeat of 5-fold cross validation
seed = 0

# define the grid search parameters
batch_size = [64, 128, 256]
n_epochs = [256, 512, 1024]
learning_rate = [0.001, 0.01, 0.1]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.1, 0.2, 0.3]
#neurons = [1, 5, 10, 15, 20, 25, 30]
n = len(batch_size)*len(n_epochs)*len(learning_rate)*len(dropout_rate)

kf = KFold(5, shuffle=True, random_state=seed)
kf.get_n_splits(df)
count = 1
score = 0

for bs in batch_size:
    for ne in n_epochs:
        for lr in learning_rate:
            for dr in dropout_rate:
                print( '{0} of {1}'.format(count,n) )
                scores = []
                for train_index, test_index in kf.split(df):
                    df_train, df_test = df.iloc[train_index], df.iloc[test_index]
                    np.random.seed(1)
                    _ = torch.manual_seed(2)
                    df_val = df_train.sample(frac=0.2)
                    #df_train = df_train.drop(df_val.index) # we do not drop the validation data from the training set

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
                    dropout = dr
                    net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)

                    model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
                    batch_size = bs

                    model.optimizer.set_lr(lr)
                    epochs = ne

                    callbacks = [tt.callbacks.EarlyStopping()]
                    verbose = False
                    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                                    val_data=val.repeat(10).cat())
                    model.partial_log_likelihood(*val).mean()

                    _ = model.compute_baseline_hazards()
                    surv = model.predict_surv_df(x_test)

                    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
                    scores.append( ev.concordance_td() )

                if np.mean(scores) > score:
                    score = np.mean(scores)
                    best_bs = bs
                    best_ne = ne
                    best_lr = lr
                    best_dr = dr
                count += 1
                print(np.mean(scores))

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% 5 repeats of 5-fold cross-validation
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

results_ci = [] #record concordance index
results_b = [] #record brier score

count = 1
for random_state in range(5):
    _ = torch.manual_seed(random_state+20)

    kf = KFold(n_splits=5,shuffle=True,random_state=random_state)
    kf.get_n_splits(df)

    for train_index, test_index in kf.split(df):
        print(f'{count} of 25')
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]
        np.random.seed(random_state+1)
        _ = torch.manual_seed(random_state+2)
        df_val = df_train.sample(frac=0.2)
        #df_train = df_train.drop(df_val.index) # we do not drop the validation data from the training set

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
        dropout = best_dr
        net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)

        model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
        batch_size = best_bs

        model.optimizer.set_lr(best_lr)
        epochs = best_ne

        callbacks = [tt.callbacks.EarlyStopping()]
        verbose = False
        log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                        val_data=val.repeat(10).cat())
        model.partial_log_likelihood(*val).mean()

        _ = model.compute_baseline_hazards()
        surv = model.predict_surv_df(x_test)

        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        results_ci.append( ev.concordance_td() )

        times = np.linspace(np.percentile(df_train.duration,25), np.percentile(df_train.duration,75), 10)
        results_b.append( ev.integrated_brier_score(times) )
        count += 1

# Print results
print( 'Average concordance index ({0} repeats of 5-fold cross validation): {1}'.format( 5, round(np.mean(results_ci),4) ) )
print( 'Standard deviation: {}'.format( round(np.std(results_ci,ddof=1),4) ) )
print( 'Average Brier score for ({0} repeats of 5-fold cross validation): {1}'.format( 5, round(np.mean(results_b),4) ) )
print( 'Standard deviation: {}'.format( round(np.std(results_b,ddof=1),4) ) )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Train final model
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Set up model
np.random.seed(111)
_ = torch.manual_seed(222)
df_val = df.sample(frac=0.2)
#df_train = df_train.drop(df_val.index) # we do not drop the validation data from the training set
x_train = x_mapper.fit_transform(df).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
labtrans = CoxTime.label_transform()
get_target = lambda df: (df['duration'].values, df['end'].values)
y_train = labtrans.fit_transform(*get_target(df))
y_val = labtrans.transform(*get_target(df_val))
val = tt.tuplefy(x_val, y_val)
val.shapes()
val.repeat(2).cat().shapes()
in_features = x_train.shape[1]
num_nodes = [32, 32]
batch_norm = True
dropout = best_dr
net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
batch_size = best_bs

model.optimizer.set_lr(best_lr)
epochs = best_ne

callbacks = [tt.callbacks.EarlyStopping()]
verbose = False
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val.repeat(10).cat())
