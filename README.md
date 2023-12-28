# COMP4388: Machine Learning - Fall 2023/2024

## Project Overview

In this individual project, the goal is to build a machine learning model that predicts whether a given person has diabetes or not. The dataset used for this project can be found [here](https://www.dropbox.com/scl/fi/ahlg01iial19mfl7wrjsy/Diabetes.csv?rlkey=7vwl95ly3hcdvqmwo7t3ply4j&dl=0). The project involves tasks related to Exploratory Data Analysis (EDA), data visualization, and the application of linear regression and k-Nearest Neighbors (kNN) classification algorithms.

## Tasks
1. **Exploratory Data Analysis (EDA):** Print the summary statistics of all attributes in the dataset.
2. **Class Label Distribution:** Display the distribution of the class label (Diabetic) and highlight any notable patterns.
3. **Age Group Analysis:** Draw histograms for each age group, detailing the number of diabetics in each subgroup.
4. **Density Plot for Age and BMI:** Visualize the density plot for the age and BMI attributes.
5. **Correlation Visualization:** Create a correlation matrix plot to visualize the relationships between features.
6. **Data Cleaning and Correlation-Based Feature Selection:** Based on the correlation matrix, decide which features to retain for machine learning models.
7. **Dataset Splitting:** Split the dataset into training (80%) and test (20%).

### Regression Tasks
1. **LR1 - Linear Regression Model:** Apply linear regression to predict the "Age" using all independent attributes.
2. **LR2 - Linear Regression Model:** Apply linear regression using the most important feature based on the correlation matrix.
3. **LR3 - Linear Regression Model:** Apply linear regression using the set of 3-most important features based on the correlation matrix.
4. **Model Comparison:** Compare the performance of LR1, LR2, and LR3 using relevant performance metrics.

### Classification Tasks
1. **kNN Classifier:** Run a k-Nearest Neighbors classifier to predict the "Diabetic" feature using the test set.
2. **kNN Models with Different k Values:** Generate kNN classifiers using various values of k, compare their performance, and include ROC/AUC score and Confusion Matrix.
3. **Results Analysis:** Report and explain the results in an appropriate table, discussing why one model outperforms the others.

