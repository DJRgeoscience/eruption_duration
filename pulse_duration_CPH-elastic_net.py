# -*- coding: utf-8 -*-

##########################################################################################################################
# Pulse - Cox Proportional Harazards Model with Ridge Penalty
##########################################################################################################################

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Import libraries
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
#%% Train final model
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
seed = 1

estimated_alphas = 10. ** np.linspace(-3, 0, 50)

cv = KFold(n_splits=5, shuffle=True, random_state=seed)
gcv = GridSearchCV(
                    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis()),
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

print( 'Results of Penalized Cox Model (elastic net) for events.' )
print( 'Average concordance index (5 repeats of 5-fold cross validation):', round(max(mean),2) )
print('Standard deviation:', round( std[ mean[mean==max(mean)].index[0] ], 2 ) )


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#%% Visalize final coefficients
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]

best_coefs = pd.DataFrame(
                            best_model.coef_,
                            index=Xt.columns,
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