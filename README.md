# Project Template: Machine Learning for Predictive Modeling (with Python)

In this file, I explain how you can solve any empirical problem with machine learning with just a few simple steps and an interest in predictive modeling. Let's get started!

# 1) Define Problem

The key is to first identify a real-world problem that is challenging to solve without empirical evidence/data (e.g. classifying different types of plants/rocks, predicting urban housing markets, etc). This problem will ideally have a significant impact on your life, and you solving this problem with machine learning should be extremely helpful to this task. In turn, when you encounter this problem in the future, you should be able to provide a reliable and accurate solution.

After the problem has been identified, the next step is to locate and store datasets of past instances of the problem at hand. A common and simple way of storing datasets is by using a CSV file. Next, you must find and install the software tools required to model your dataset via for making predictions. In my projects, I use the Python programming language and its Scikit-learn, SciPy, NumPy, Pandas, and Matplotlib libraries. Afterwards, create a code file in your IDE of choice (e.g. Atom), and place your project in a folder that includes the file and your dataset. In the code file, import the libraries and load the dataset that you will be using.

# 2) Summarize Data

Now that you have set up our machine learning project, you are ready to dive deep into the data! Before you can build your machine learning models, you must first understand the data that you will be using for your problem. The two key ways of summarizing your data are through descriptive statistics and data visualizations.

Descriptive statistics can provide basic insights about your dataset. Before computing descriptive statistics, it can be helpful to first view the dimensions, first 'k' data entries, data types for the dataset. Descriptive statistics can you help learn more about the dataset by calculating the range, mean & standard deviation, & median with percentiles for each attribute; all of this information can be provided as a results report. You can also calculate the skew for each attribute along with the correlation between each attribute. For classification problems, viewing the distribution across each class can also indicate if special data handling techniques will be required.

Data visualizations can greatly supplement descriptive statistics by concisely representing the distribution of thousands of data entries in a single plot. For univariate plots, histograms, density plots, and boxplots work best. For multivariate plots, correlation matrix plots and scatter plot matrices can be extremely effective. For gaining the best understanding of your data, I recommend you use multiple plot types for and across each attribute.

# 3) Prepare Data

Now that you have a better understanding of your dataset, it's time to prepare the dataset in order to best expose its distribution to solving our problem using the given input attributes and resulting output variable. Data preparation can be broken down into three sub-steps: data cleaning, feature selection, and data transformation. It can be helpful to move between  this step ("Prepare Data") and the next step ("Evaluate Algorithms") multiple times until your model achieves a sufficient level of accuracy with making predictions on new data.

Because real-world data can be a bit messy, it's important to first clean your dataset by: removing duplicates, marking missing values, and inputing replacements for the missing values. If your data isn't cleaned, then it will be difficult to build accurate machine learning models on new data.

With your newly cleaned data, feature selection must be performed to optimize the predictive accuracy of your machine learning model. Effective techniques for feature selection include: univariate selection, recursive feature elimination (RFE), principle component analysis (PCA), and feature importance. Without feature selection, your model becomes susceptible to overvaluing irrelevant or partially relevant attributes.

Lastly, you should transform your data to best fit your problem based on its distribution and attributes. To perform a data transform, you will: (1) load your dataset, (2) split the dataset into input and output variables, (3) apply a pre-processing algorithm to the variables, and (4) report the results of your data. Data transforms include: rescaling, standardizing, normalizing, and binarizing the dataset.

# 4) Evaluate Algorithms

The majority of your effort for your machine learning project should focus on this step (along with the prior step of "Prepare Data") as the core of your machine learning model, along with its predictive accuracy, will rely on which machine learning algorithms you select and implement. The steps of the algorithm evaluation process are: (1) defining test options for dataset splitting, (2) selecting evaluation metrics for algorithm performance, and (3) spot-checking multiple linear and non-linear machine learning (classification or regression) algorithms. After doing this, be sure to compare the estimated accuracy of each algorithm to determine which ones will work best for you to solve your problem.

There are many ways in which you can split your dataset to best test your data. Some of these techniques include: train & test sets, k-fold cross-validation, leave one out cross-validation, and repeated random test-train splits. When in doubt, it's generally best to choose k-fold cross validation with k=10.

Elavuation metrics serve as KPIs for a given algorithms's predictive accuracy. For classification problems, metrics include: classification accuracy, logistic loss, area under ROC (rate of change) curve, confusion matrix, and classification report. For regression problems, metrics include: mean absolute error (MAE), mean squared error (MSE), and coefficient of determination (R^2).

While you likely won't need to implement machine learning algorithms from scratch for your project, it is important to distinguish between which algorithms you should use to solve your problem. Classification algorithms include: logistic regression, linear discriminant analysis (LDA), k-Nearest Neighbors (kNN), naive bayes, decision trees, and support vector machines (SVM). Regression algorithms include: linear regression, ridge regression, LASSO regression, elastic net regression, KNN, decision trees, and SVM.

# 5) Improve Accuracy

If your machine learning model has a predictive accuracy sufficient to solve your problem, that's great! However, if it doesn't, or you simply require the most accurate predictive model possible, then it's critical you improve your model's accuracy. First, you should search for and determine which combination of paremeters leads to the highest accuracy for each algorithm in your model. Then, you should use one or both of the two primary techniques for improving model accuracy are ensembles and algorithm tuning.

For ensembles, the goal is to combine the predictions from different models and create a united ensemble prediction. Different ensemble techniques include: bagging (e.g. bagged decision trees, random forests, extra trees), boosting (AdaBoost, stochastic gradient boosting), and voting.

Algorithm tuning (or hyperparameter optimization) improves your model's accuracy by searching for parameters that will maximize your model's scalability and overall success. The two primary techniques for tuning algorithms are grid search parameter tuning and random search parameter tuning.

# 6) Finalize Model

The final step in completing your machine learning project is finalizing your optimized model. This step typically entails: testing your predictive model on unseen data and saving your model to a file for later use.

# 7) Present Results

Congratulations! You are have now completed your very own machine learning project! There are a number of ways that you can share with the world what you've accomplished and learned through your project. One scalable method of sharing machine learning projects on the internet is by uploading your project's code file along with a description of the project to GitHub. From there, you can make your project publicly available, so anyone can view your project's summary and details and learn how you went through solving your problem with machine learning. Best of luck!

# Credits:
The inspiration behind my machine learning projects is from "Machine Learning Mastery with Python" by Jason Brownlee.
