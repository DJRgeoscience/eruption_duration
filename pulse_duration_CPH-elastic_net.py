# -*- coding: utf-8 -*-

##########################################################################################################################
# Pulse - Cox Proportional Harazards Model with Ridge Penalty
##########################################################################################################################

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Import libraries
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, RepeatedKFold, cross_val_score
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

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Prepare data
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# remove highly correlated (r >= ~0.7) features
remove = [ 'rift', 'intraplate', 'ctcrust1', 'meanslope', 'shield']
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

alphas = 10. ** np.linspace(-5, 0, 10)
cox_elastic_net = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=alphas, normalize=True)

cox_elastic_net.fit(Xt, y)

coefficients_elastic_net = pd.DataFrame(
                                        cox_elastic_net.coef_,
                                        index=Xt.columns,
                                        columns=np.round(cox_elastic_net.alphas_, 5)
                                       )

plot_coefficients(coefficients_elastic_net, n_highlight=5)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Optimization
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
seed = 1

estimated_alphas = 10. ** np.linspace(-3, 0, 50)

cv = KFold(n_splits=5, shuffle=True, random_state=seed)
gcv = GridSearchCV(
                    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.5)),
                    param_grid={"coxnetsurvivalanalysis__alphas": [[a] for a in estimated_alphas]},
                    cv=cv,
                    error_score=0.5,
                    n_jobs=-1
                  ).fit(Xt, y)

cv_results = pd.DataFrame(gcv.cv_results_)

alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
mean = cv_results.mean_test_score
std = cv_results.std_test_score

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(alphas, mean)
ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
ax.set_xscale("log")
ax.set_ylabel("concordance index")
ax.set_xlabel("alpha")
ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
ax.axhline(0.5, color="grey", linestyle="--")
ax.grid(True)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Cross validation, concordance index
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Initialize random number generator
seed = 1

kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)

model = CoxnetSurvivalAnalysis(l1_ratio=0.5,n_alphas=1,alphas=gcv.best_params_["coxnetsurvivalanalysis__alphas"],fit_baseline_model=True)
results = cross_val_score(model, Xt, y, cv=kf)

# Print results
print( 'Average concordance index for ({0} repeats of 5-fold cross validation): {1}'.format( 5, round(results.mean(),4) ) )
print( 'Standard deviation: {}'.format( round(results.std(ddof=1),4) ) )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Cross validation, integrated brier score
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

random_states = [11,12,13,14,15]
results = []
model = CoxnetSurvivalAnalysis(l1_ratio=1,n_alphas=1,alphas=gcv.best_params_["coxnetsurvivalanalysis__alphas"],fit_baseline_model=True)
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
        times = times = np.linspace( np.percentile([time[1] for time in y_test], 25), np.percentile([time[1] for time in y_test], 75), 10 )
        preds = np.asarray( [ [sf(t) for t in times] for sf in survs ] )
        score = integrated_brier_score(y_train, y_test, preds, times)

        results.append( score )

# Print results
print( 'Average Brier score for ({0} repeats of 5-fold cross validation): {1}'.format( len(random_states), round(np.mean(results),4) ) )
print( 'Standard deviation: {}'.format( round(np.std(results,ddof=1),4) ) )

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Train final model
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

model = CoxnetSurvivalAnalysis(l1_ratio=1,n_alphas=1,alphas=gcv.best_params_["coxnetsurvivalanalysis__alphas"],fit_baseline_model=True)
model.fit(Xt,y)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Calculate shap values
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

explainer = shap.Explainer(model.predict, Xt, feature_names=feature_names)
shaps = explainer(Xt)
shap.summary_plot(shaps, Xt)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Alternative method for evaluating features
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]

best_coefs = pd.DataFrame(
                            best_model.coef_,
                            index=feature_names,
                            columns=["coefficient"]
                         )

non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
print("Number of non-zero coefficients: {}".format(non_zero))

non_zero_coefs = best_coefs.query("coefficient != 0")
coef_order = non_zero_coefs.abs().sort_values("coefficient").index

_, ax = plt.subplots(figsize=(6, 8))
non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
ax.set_xlabel("coefficient")
ax.grid(True)
