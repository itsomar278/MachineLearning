import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc



# Load the dataset using double backslashes
url = "C:\\Users\\omar\\Downloads\\Diabetes.csv"
df = pd.read_csv(url)

def print_data_summary(dataframe):
    print(dataframe.info())
    print(dataframe.describe())

def count_and_replace_missing_values(dataframe):
    last_column = dataframe.columns[-1]
    dataframe.loc[:, dataframe.columns != last_column] = dataframe.loc[:, dataframe.columns != last_column].replace(0, pd.NA)
    missing_values_count = dataframe.isna().sum()

    # Print the count of missing values in each column
    print("Count of missing values (NA's) in each column after replacing 0's:")
    for column, count in missing_values_count.items():
        print(f"{column}: {count}")


def class_label_distribution(dataframe, target):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target, data=dataframe)
    plt.title('Class Label Distribution')

    # Indicate highlights
    diabetic_count = dataframe[target].sum()
    non_diabetic_count = len(dataframe) - diabetic_count
    plt.text(0, diabetic_count, f'Diabetic: {diabetic_count}', ha='center', va='bottom')
    plt.text(1, non_diabetic_count, f'Non-Diabetic: {non_diabetic_count}', ha='center', va='bottom')

    plt.show()

def histogram_by_age_group():
    plt.figure(figsize=(10, 6))


    bins = [20, 30, 40, 50, 60, 70, 80, 90]
    labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']
    df['AgeGroup'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)

    sns.histplot(data=df, x='AgeGroup', hue='Diabetic', multiple='stack', palette='Set2', edgecolor='w')
    plt.title('Distribution of Diabetics in Each Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.show()

def density_plot_age(dataframe, age_col):
    """Show the density plot for the specified age column."""
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=dataframe[age_col], fill=True)
    plt.title(f'Density Plot for {age_col}')
    plt.xlabel(age_col)
    plt.ylabel('Density')
    plt.show()

def handle_missing_values(dataframe):
    """Fill NA's with column means."""
    dataframe = dataframe.fillna(dataframe.mean())
    return dataframe

def plot_correlation_matrix(dataframe):
    """Visualize the correlation matrix between all features."""
    correlation_matrix = dataframe.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def linear_regression_to_learn_age(dataframe):
    """Apply linear regression to learn the attribute 'Age' using all independent attributes."""
    # Separate features (independent variables) and target variable
    X = dataframe.drop(['AGE'], axis=1)  # All independent attributes except 'AGE'
    y = dataframe['AGE']  # Target variable is 'AGE'

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the linear regression model
    lr_model = LinearRegression()

    # Fit the model to the training data
    lr_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lr_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the model's performance metrics
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    return lr_model  # Return the trained linear regression model

def linear_regression(dataframe, independent_columns, target_column):
    if len(independent_columns) == 1:
        X = dataframe[independent_columns].values.reshape(-1, 1)
    else:
        X = dataframe[independent_columns].values

    y = dataframe[target_column]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the linear regression model
    lr_model = LinearRegression()

    # Fit the model to the training data
    lr_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lr_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the model's performance metrics
    print(f'Linear Regression with {", ".join(independent_columns)} as independent variables')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    return lr_model  # Return the trained linear regression model

def run_knn_classifier(dataframe, features, target, test_size=0.2, random_state=42, n_neighbors=3):
    """
    Run a k-Nearest Neighbors classifier and evaluate its performance.

    Parameters:
    - dataframe: Pandas DataFrame containing features and the target variable.
    - features: List of feature column names.
    - target: Name of the target variable column.
    - test_size: The proportion of the dataset to include in the test split.
    - random_state: Seed used by the random number generator for reproducibility.
    - n_neighbors: Number of neighbors to use for k-NN.

    Returns:
    - None
    """
    # Separate features (independent variables) and target variable
    X = dataframe[features]
    y = dataframe[target]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the k-NN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the model to the training data
    knn_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn_classifier.predict(X_test)

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, knn_classifier.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Print the evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

    # Print ROC curve and AUC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve (n={n_neighbors})')
    plt.legend(loc='lower right')
    plt.show()



# Call the functions
print_data_summary(df)
count_and_replace_missing_values(df)
class_label_distribution(df, 'Diabetic')
histogram_by_age_group()
df.drop(columns=['AgeGroup'], inplace=True)
density_plot_age(df, 'AGE')
density_plot_age(df, 'BMI')
df = handle_missing_values(df)
count_and_replace_missing_values(df)
plot_correlation_matrix(df)
lr1 = linear_regression_to_learn_age(df)
lr2 =  linear_regression(df, ['NPG'], 'AGE')
lr3 = linear_regression(df, ['NPG', 'PGL', 'DIA'], 'AGE')
run_knn_classifier(df, df.columns[:-1], 'Diabetic' , n_neighbors=1)
run_knn_classifier(df, df.columns[:-1], 'Diabetic' , n_neighbors=7)
run_knn_classifier(df, df.columns[:-1], 'Diabetic' , n_neighbors=14)
run_knn_classifier(df, df.columns[:-1], 'Diabetic' , n_neighbors=60)









