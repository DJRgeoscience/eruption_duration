# -*- coding: utf-8 -*-

##########################################################################################################################
# Event - Cox Proportional Harazards Model with Ridge Penalty
##########################################################################################################################

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Import libraries
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

from sksurv.linear_model import CoxPHSurvivalAnalysis as CPHSA
from sksurv.metrics import integrated_brier_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score, GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(
            alpha_min, coef, name + "   ",
            horizontalalignment="right",
            verticalalignment="center"
        )

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")

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
feature_names = Xt.columns.tolist()

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Visualize penalty effect on coefficients
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

alphas = 10. ** np.linspace(-4, 4, 50)
coefficients = {}

cph = CPHSA()
for alpha in alphas:
    cph.set_params(alpha=alpha)
    cph.fit(Xt, y)
    key = round(alpha, 5)
    coefficients[key] = cph.coef_

coefficients = (pd.DataFrame
    .from_dict(coefficients)
    .rename_axis(index="feature", columns="alpha")
    .set_index(Xt.columns))

plot_coefficients(coefficients, n_highlight=5)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Optimize model
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
seed = 1

alphas = 10. ** np.linspace(-5, 16, 100)
cv = KFold(n_splits=5, shuffle=True, random_state=seed)
gcv = GridSearchCV(
                    make_pipeline(StandardScaler(), CPHSA()),
                    param_grid={"coxphsurvivalanalysis__alpha": [a for a in alphas]},
                    cv=cv,
                    error_score=0.5,
                    n_jobs=-1
                  ).fit(Xt, y)

cv_results = pd.DataFrame(gcv.cv_results_)

mean = cv_results.mean_test_score.values
std = cv_results.std_test_score.values

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(alphas, mean)
ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
ax.set_xscale("log")
ax.set_ylabel("concordance index")
ax.set_xlabel("alpha")
ax.axvline(gcv.best_params_["coxphsurvivalanalysis__alpha"], c="C1")
ax.axhline(0.5, color="grey", linestyle="--")
ax.grid(True)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Cross validation, concordance index
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Initialize random number generator
seed = 10

kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)

model = CPHSA(alpha=gcv.best_params_["coxphsurvivalanalysis__alpha"])
results = cross_val_score(model, Xt, y, cv=kf)

# Print results
print( 'Average concordance index for ({0} repeats of 5-fold cross validation): {1}'.format( 5, round(results.mean(),4) ) )
print( 'Standard deviation: {}'.format( round(results.std(ddof=1),4) ) )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Cross validation, integrated brier score
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

random_states = [11,12,13,14,15]
results = []
model = CPHSA(alpha=gcv.best_params_["coxphsurvivalanalysis__alpha"])
Xt = Xt.values

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
        times = np.linspace( np.percentile([time[1] for time in y_test], 25), np.percentile([time[1] for time in y_test], 75), 10 )
        preds = np.asarray( [ [sf(t) for t in times] for sf in survs ] )
        score = integrated_brier_score(y_train, y_test, preds, times)

        results.append( score )

# Print results
print( 'Average Brier score for ({0} repeats of 5-fold cross validation): {1}'.format( len(random_states), round(np.mean(results),4) ) )
print( 'Standard deviation: {}'.format( round(np.std(results,ddof=1),4) ) )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Train final model
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

model = CPHSA(alpha=gcv.best_params_["coxphsurvivalanalysis__alpha"])
model.fit(Xt,y)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Calculate shap values
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

explainer = shap.Explainer(model.predict, Xt, feature_names=feature_names)
shaps = explainer(Xt)
shap.summary_plot(shaps, Xt)
