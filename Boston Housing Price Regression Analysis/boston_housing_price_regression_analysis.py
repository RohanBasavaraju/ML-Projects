# Boston Housing Price Regression Analysis - Inspired by "Machine Learning Mastery with Python" by Jason Brownlee

# 1) Prepare Problem

# Load libraries
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
filename = 'housing.csv'
names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
dataset = read_csv(filename, delim_whitespace=True, names=names)

# 2) Summarize Data

# Descriptive statistics
print(dataset.shape) # Dimensions
print(dataset.dtypes) # Data types for each attribute
print(dataset.head(20)) # Peek at first 20 data entries -> will need to do data transforms later on
set_option('precision', 1)
print(dataset.describe()) # Mean, median, range, etc for each attribute -> will need to rescale data
set_option('precision', 2)
print(dataset.corr(method='pearson')) # Correlation (-1<=R<=1) between attributes -> many have strong correlations

# Data visualizations

# Univariate Plots
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1) # Histogram
pyplot.show() # Some attributes may have exponential or binomial distributions
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=False, fontsize=1) # Density Plot
pyplot.show() # Some attributes may have exponential or binomial distributions
              # -> NOX, RM, LSTAT may have skewed Gaussian distributions
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=8) # Boxplot
pyplot.show() # Many attributes may have skewed distributions -> many outliers

#Multivariate Plots
scatter_matrix(dataset) # Scatter Plot Matrix
pyplot.show() # Some highly correlated attributes have good structure
              # -> predictable curved (not linear) relationships
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none') # Correlation Matrix
fig.colorbar(cax)
ticks = numpy.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show() # Some attributes share strong correlations -> may need to remove them

# 3) Prepare Data

'''
Because the dataset is very structured, we can identify some data preparation
techniques to best expose the dataset's structure for improved model accuracy.

As machine learning projects will typically revisit steps 3 and 4 multiple times,
the code for step 3 has been provided at the end of step 4. This way, the code
for step 3 follows the logical flow of our project.
'''

# Data Cleaning
'''
Because our dataset has been pre-cleaned, there's no need to clean our dataset.
However, if we were using a raw dataset, the data would likely be messy, so we
would need to clean our dataset.
'''

# Feature Selection
'''
Feature Selection: remove the most strongly correlated attributes
'''

# Data Transforms
'''
Data Normalization: reduce the effects of different scales
Data Standardization: reduce the effects of different distributions
'''

# 4) Evaluate Algorithms

# Split-out validation dataset
array = dataset.values
x = array[:,0:13]
y = array[:,13]
validation_size = 0.2
seed = 7
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
num_folds = 10 # 10-fold cross validation
seed = 7
scoring = 'neg_mean_squared_error' # Mean Squared Error (MSE)

# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression())) # Linear Regression
models.append(('LASSO', Lasso())) # Lasso Regression
models.append(('EN', ElasticNet())) # ElasticNet
models.append(('KNN', KNeighborsRegressor())) # k-Nearest Neighbors
models.append(('DT', DecisionTreeRegressor())) # Decision Tree
models.append(('SVR', SVR())) # Support Vector Machines

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(name + ': ', cv_results.mean(), cv_results.std())
    # LR and DT have the 1st and 2nd lowest MSE values

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results) # Boxplot comparing MSE median & distribution for each model
ax.set_xticklabels(names)
pyplot.show() # Similar distributions across models except smaller distribution for DT
# Differing scales of models may hurt performance -> should standardize data

# Data Standardization (continuation of Step 3)
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))

results = []
names = []
for name, model in pipelines:
  kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  print(name + ': ', cv_results.mean(), cv_results.std())
  # Scaling the raw data resulted in a significant reduction of KNN's MSE values
  # -> KNN seems to now be the most accurate model

# Compare (Post-Data Standardization) Algorithms

fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results) # boxplot of each model's median MSE & its MSE distribution
ax.set_xticklabels(names)
pyplot.show() # KNN has a small distribution and the lowest median MSE

# 5) Improve Accuracy

# Algorithm Tuning
scaler = StandardScaler().fit(x_train) # fit data to standard scale
rescaledx = scaler.transform(x_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21]) # find optimal k
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor() # tune KNN regression algorithm
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True) # 10-fold cross-validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledx, y_train) # fit KNN with standardized input data

print('Best: ', grid_result.best_score_, 'using ', grid_result.best_params_)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(mean, (stdev), 'with: ', param)

# Ensembles
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())]))) # Boosting: AdaBoost
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())]))) # Boosting: Gradient Boosting
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor(n_estimators=10))]))) # Bagging: Random Forests
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor(n_estimators=10))]))) # Bagging: Extra Trees

results = []
names = []
for name, model in ensembles:
  kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  print(name + ': ', cv_results.mean(), cv_results.std())
  # MSE values were lower (i.e. better) with ensembles than with linear & nonlinear algorithms

# Compare Ensemble Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results) # boxplot of each model's median MSE & its MSE distribution
ax.set_xticklabels(names)
pyplot.show() # GBM has highest mean while ET has highest median -> both: similar distributions

# GBM Algorithm Tuning
scaler = StandardScaler().fit(x_train) # fit data to standard scale
rescaledx = scaler.transform(x_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400])) # find optimal n_estimators
model = GradientBoostingRegressor(random_state=seed) # tune GBM Boosting Ensemble
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True) # 10-fold cross validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, iid=True)
grid_result = grid.fit(rescaledx, y_train) # fit GBM with standardized input data

print('Best: ', grid_result.best_score_, 'using ', grid_result.best_params_)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(mean, (stdev), 'with: ', param)
# Best configurations was n_estimators=400 -> 0.65 units better than untuned method

# 6) Finalize Model

# Prepare the model
scaler = StandardScaler().fit(x_train)
rescaledx = scaler.transform(x_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledx, y_train)

# Transform the validation dataset
rescaledValidationx = scaler.transform(x_test)
predictions = model.predict(rescaledValidationx)
print(mean_squared_error(y_test, predictions)) # Estimated MSE=11.8 -> close to (-)9.3
