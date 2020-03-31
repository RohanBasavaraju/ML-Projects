# Geologic Materials Detection & Classification - Inspired by "Machine Learning Mastery with Python" by Jason Brownlee

# 1) Prepare Problem

# Load libraries
import numpy
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Load dataset
url = 'sonar.all-data.csv'
dataset = read_csv(url, header=None)

# 2) Summarize Data

# Descriptive statistics
print(dataset.shape) # Dimensions
set_option('display.max_rows', 500)
print(dataset.dtypes) # Data types for each attribute
set_option('display.width', 100)
print(dataset.head(20)) # Peek at first 20 data entries for each attribute
# Data has the same scale & the 'class' attribute has a string data type
set_option('precision', 3)
print(dataset.describe()) # Mean, median, range, & distribution for each attribute
# Data has same range (as expected) but differing mean values -> may want to standardize data
print(dataset.groupby(60).size()) # Class distribution -> classes are fairly balanced

# Data visualizations

# Univariate Plots
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1) # Histogram
pyplot.show() # Many attributes have Gaussian distributions & some have exponential distributions
dataset.plot(kind='density', subplots=True, layout=(8,8), sharex=False, legend=False, fontsize=1) # Density Plot
pyplot.show() # Many attributes have skewed distributions -> can correct skewness using a power transform (e.g. Box-Cox)

# Multivariate Plots
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none') # Correlation Matrix
fig.colorbar(cax)
pyplot.show() # Strong structure in attribute order -> makes sense due to sonar's angle of sensors
              # -> attributes closer together are more positively correlated
              # -> attributes farther apart are more negatively correlated

# 3) Prepare Data
'''
Data preparation will take place at the end of step 4 by creating a data pipeline
to fit, transform, and standardize the datset.
'''

# 4) Evaluate Algorithms

# Split-out validation dataset
array = dataset.values
x = array[:,0:60].astype(float)
y = array[:,60]
validation_size = 0.20 # 20% of dataset for testing & 80% for model training
seed = 7
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
num_folds = 10 # 10-fold cross validation
seed = 7
scoring = 'accuracy' # evaluation metric

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear'))) # Logistic Regression
models.append(('LDA', LinearDiscriminantAnalysis())) # Linear Discriminant Analysis
models.append(('KNN', KNeighborsClassifier())) # k-Nearest Neighbors
models.append(('DT', DecisionTreeClassifier())) # Decision Tree
models.append(('NB', GaussianNB())) # Naive Bayes
models.append(('SVM', SVC(gamma='auto'))) # Support Vector Machines

results = []
names = []
for name, model in models:
  kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  print((name, cv_results.mean(), cv_results.std()))
  # k-Nearest Neighbors & Logistic Regression had the 1st & 2nd most accurate models

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results) # Boxplot of each model's median accuracy & its distribution
ax.set_xticklabels(names)
pyplot.show() # KNN has low variance
# SVM has surprisingly poor results -> should standardize data for best results

# Dataset Standardization (Step 3)
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression(solver='liblinear'))])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC(gamma='auto'))])))

results = []
names = []
for name, model in pipelines:
  kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  print(name, cv_results.mean(), cv_results.std())
  # SVM now has the most accurate model & KNN performance has also improved

# Compare Algorithms of Standardized Data
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results) # Boxplot of each model's median accuracy & its distribution
ax.set_xticklabels(names)
pyplot.show() # SVM & KNN have the highest median accuracy with lower variance

# 5) Improve Accuracy

# Algorithm Tuning

# KNN Tuning
scaler = StandardScaler().fit(x_train)
rescaledx = scaler.transform(x_train)
neighbors = [1,3,5,7,9,11,13,15,17,19,21] # Find optimal number of neighbors
param_grid = dict(n_neighbors=neighbors)
model = KNeighborsClassifier() # KNN Classification
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True) # 10-fold cross validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) # Grid Search Tuning
grid_result = grid.fit(rescaledx, y_train)
print((grid_result.best_score_, grid_result.best_params_)) # Find score associated with the best parameter

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print((mean, stdev, param)) # Optimal neighbor size of 1 ->
                                # KNN should only use closest neighbor for making predictions

# SVM Tuning
scaler = StandardScaler().fit(x_train)
rescaledx = scaler.transform(x_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0] # Find optimal C value for optimal kernel
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid'] # Find optimal C value for optimal kernel
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC(gamma='auto') # SVM Classification
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True) # 10-fold cross validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) # Grid Search Tuning
grid_result = grid.fit(rescaledx, y_train)
print((grid_result.best_score_, grid_result.best_params_)) # Find score associated with the best parameter

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print((mean, stdev, param)) # C = 1.5 for kernel = 'rbf'

# Ensembles
ensembles = []
ensembles.append(('AB', AdaBoostClassifier())) # AdaBoost Boosting Ensemble
ensembles.append(('GBM', GradientBoostingClassifier())) # Gradient Boosting Ensemble
ensembles.append(('RF', RandomForestClassifier(n_estimators=10))) # Random Forest Bagging Ensemble
ensembles.append(('ET', ExtraTreesClassifier(n_estimators=10))) # Extra Trees Bagging Ensemble

results = []
names = []
for name, model in ensembles:
  kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True) # 10-fold cross validation
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  print(name, cv_results.mean(), cv_results.std()) # Boosting techniques have higher mean accuracy

# Compare Ensembles
fig = pyplot.figure()
fig.suptitle('Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results) # Boxplot of each ensemble's median accuracy & its distribution
ax.set_xticklabels(names)
pyplot.show() # Gradient Boosting Ensemble Shows Highest Potential with >97% accuracy

# 6) Finalize Model

# Model Preparation
scaler = StandardScaler().fit(x_train) # Standardized Data Fitting
rescaledx = scaler.transform(x_train) # Standardized Data Transform
model = SVC(C=1.5) # SVM Classification
model.fit(rescaledx, y_train) # Build SVM classification model

rescaledValidationx = scaler.transform(x_validation)
predictions = model.predict(rescaledValidationx)
print(accuracy_score(y_validation, predictions)) # Model Accuracy
print(confusion_matrix(y_validation, predictions)) # Confusion Matrix
print(classification_report(y_validation, predictions)) # Classification Report

print('We are now complete with our machine learning project!')
