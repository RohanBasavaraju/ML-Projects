# Iris Flowers Classification - Inspired by "Machine Learning Mastery with Python" by Jason Brownlee

# 1) Prepare Problem

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
filename = 'iris.data.csv'
names = ['sepal-length','sepal-width','pedal-length','pedal-width','class'] # attributes
dataset =  read_csv(filename, names=names) # dataframe

# 2) Summarize Data

# Descriptive statistics
print(dataset.shape) # dimensions
print(dataset.head(20)) # first 20 instances of dataset
print(dataset.describe()) # mean, SD, median, percentiles, min/max for each attribute
print(dataset.groupby('class').size()) # class distribution

# Data visualizations

#Univariate Plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) # boxplot for each attribute
pyplot.show()
dataset.hist() #histogram for each attribute -> check for Gaussian distributions
pyplot.show()

#Multivariate Plots
scatter_matrix(dataset) # scatter plot matrix for all attributes -> check for (linear) separation
pyplot.show()

# 3) Prepare Data
'''
As our dataset is pre-cleaned and optimized for building models, we can continue
without manually completing: data cleaning, feature selection, data transforms.
However, this is very rare and can't be ignored when working with raw data!
'''

# 4) Evaluate Algorithms

# Split-out validation dataset
array = dataset.values
x = array[:,0:4]
y = array[:,4]
validation_size = 0.2 # 80% of the dataset is for training & 20% is for testing
seed = 7
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
'''
We will use a 10-fold cross validation to estimate model accuracy on unseen data.
The code for this will be provided in the next step as it will be applied to
every algorithm that we spot check.
'''

# Spot Check Algorithms
'''
From the data visualization plots generated in step 2, we gathered that there may
be a partially linear relationship between some of the attributes. This indicates
that the six classification algorithms that we use below should provide good results.
Using varied algorithms is especially helpful as some of them are linear and others
are nonlinear.
'''
models = []
models.append(('LR',LogisticRegression(solver='liblinear', multi_class='ovr'))) # Logistic Regression
models.append(('LDA', LinearDiscriminantAnalysis())) # Linear Discriminant Analysis
models.append(('KNN', KNeighborsClassifier())) # k-Nearest Neighbors
models.append(('DT', DecisionTreeClassifier())) # Decision Tree
models.append(('NB', GaussianNB())) # Naive Bayes
models.append(('SVM', SVC(gamma='auto'))) # Support Vector Machines

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True) # 10-fold cross validation
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    # accuracy estimations obtained from 10-fold cross validation
    results.append(cv_results)
    names.append(name)
    print((name + ': ', cv_results.mean(), cv_results.std())) # print the mean & SD for each model

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithms Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results) # create a boxplot to visually evaluate each model's estimated accuracy
ax.set_xticklabels(name)
pyplot.show() # KNN and SVM had the 1st and 2nd heighest medians of estimated accuracy

# 5) Improve Accuracy
'''
As our KNN model achieved a sufficiently high estimated accuracy, we do not need
to improve the accuracy of our model using ensembles or tuning. However, this
will not occur too often, so make sure to do this step when necessary!
'''

# 6) Finalize Model

# Predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(x_train, y_train) # fit KNN model using the 80% of our dataset reserved for training
predictions = knn.predict(x_test) # predict accuracy using the 20% of our dataset reserved for testing
print(accuracy_score(y_test, predictions)) # compare predictions accuracy to expected values
print(confusion_matrix(y_test, predictions)) # represent model accuracy using confusion matrix
print(classification_report(y_test, predictions)) # represent model accuracy using classification report


print('Our machine learning project is now complete!')
