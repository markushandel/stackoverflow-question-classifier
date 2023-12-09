import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from models.gaussian_nb import GaussianNBModel
from models.logistric_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.neural_network import KerasNeuralNetworkModel
from src import preprocess_data, filter_dataframe_on_languages, filter_dataframe_on_size, load_data_sets, ucb1
from feature_selectors import UnivariateFeatureSelector, PSOFeatureSelector
from text_processors import TfidfProcessor, BowDataProcessor, Word2VecDataProcessor, BertDataProcessor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def convert_to_binary(df, label_column, specified_class):
    """
    Converts a multiclass dataset into a binary dataset by modifying the label column.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    label_column (str): The name of the column containing the labels.
    specified_class (str): The class to be considered as one binary class.

    Returns:
    pd.DataFrame: The modified dataframe with binary classification.
    """
    # Overwrite the label column for binary classification
    df[label_column] = df[label_column].apply(lambda x: specified_class if x == specified_class else 'Other')

    return df

def find_common_strings(df, column_name='body'):
    if column_name not in df.columns:
        print(f"'{column_name}' column not found in the dataframe")
        return set()

    # Split the first row into a set of strings
    common_strings = set(df.iloc[0][column_name].split())

    # Intersect with sets of strings from other rows
    for i in range(1, len(df)):
        current_strings = set(df.iloc[i][column_name].split())
        common_strings.intersection_update(current_strings)

        # If common_strings is empty, no need to proceed further
        if not common_strings:
            break

    return common_strings

def get_data():
    folder_path = "data/raw"

    df = load_data_sets(folder_path)
    # Usage example
    common_strings = find_common_strings(df)
    print("Common strings:", common_strings)


    df = preprocess_data(df)

    df = df.sample(frac=1).reset_index(drop=True)

    specified_class = 'valid-question'  # Example class to be isolated
    # df = convert_to_binary(df, 'label', specified_class)

    df = filter_dataframe_on_size(df, 300, 300)

    # print all different labels
    print(df['label'].unique())

    # print the number of rows for each label
    print(df['label'].value_counts())

    # print the number of rows for each label as a percentage of the total
    print(df['label'].value_counts(normalize=True))

    # remove the rows that contain the label 'duplicate'
    df = df[df['label'] != 'Duplicate']
    df = df[df['label'] != 'Needs more focus']

    print("POST")

    print(df['label'].value_counts(normalize=True))

    print(df['label'].unique())


    # If you want to see the counts of how many times each text is duplicated
    label_encoder = LabelEncoder()


    y = label_encoder.fit_transform(df['label'])
    X = df.drop('label', axis=1)
    return X, y

def data_processor(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # data_processor = TfidfProcessor()
    data_processor = BowDataProcessor()
    # data_processor = Word2VecDataProcessor()
    # data_processor = BertDataProcessor()

    print("Fitting data processor...")
    print(X_train.shape)
    
    data_processor.fit(X_train)
    X_train = data_processor.transform(X_train)
    X_test = data_processor.transform(X_test)

    # Create a VarianceThreshold selector with a threshold of 0 (to find constant features)
    numeric_df = pd.DataFrame(X_train).select_dtypes(include=[np.number])

    selector = VarianceThreshold(threshold=0)

    # Fit the selector to your data
    selector.fit(numeric_df)

    # Get the indices of constant features
    constant_features_indices = [i for i, var in enumerate(selector.variances_) if var == 0]

    # Get the names of constant features
    constant_features_names = [numeric_df.columns[i] for i in constant_features_indices]

    # Print constant feature names
    print("Constant numeric features:")
    print(constant_features_names)
    # return X_train, X_test, y_train, y_test

    # Initialize your feature selector
    feature_selector = UnivariateFeatureSelector(k=1000)  # or any other selector
    # feature_selector = PSOFeatureSelector(n_particles=5, n_iterations=5)

    # Fit the selector on the train features and labels
    feature_selector.fit(X_train, y_train)

    # Transform the test data (do not fit the selector again)
    X_train_selected = feature_selector.transform(X_train)
    X_test_selected = feature_selector.transform(X_test)
    return X_train_selected, X_test_selected, y_train, y_test

def train_models(X_train, X_test, y_train, y_test, models):
    n_models = len(models)
    model_counts = np.zeros(n_models)
    model_rewards = np.zeros(n_models)
    model_instances = [model() for model in models]  # Instantiate all models

    n_iterations = max(10, n_models)  # Ensure at least one iteration per model

    for iteration in range(n_iterations):
        model_index = ucb1(iteration + 1, model_counts, model_rewards)
        model = model_instances[model_index]

        model.train(X_train, y_train)
        test_predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, test_predictions)

        model_counts[model_index] += 1
        model_rewards[model_index] += accuracy

        print(f"Iteration {iteration + 1}, Model {model_index}, Accuracy: {accuracy}")

    top_model_index = np.argmax(model_rewards / model_counts)
    best_model = model_instances[top_model_index]
    average_score = model_rewards[top_model_index] / model_counts[top_model_index]

    return best_model, average_score


def plot_results(y_test, test_predictions):
    # Assuming you have your test labels (y_test) and predictions (predictions)
    cm = confusion_matrix(y_test, test_predictions)

    # Plotting the confusion matrix
    sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    print("-" * 50)

def main():
    X, y = get_data()
    print("Data loaded!")

    X_train, X_test, y_train, y_test = data_processor(X, y)
    print("Data processed!")

    # Train the model
    models = [
        KerasNeuralNetworkModel,
        LogisticRegressionModel,
        RandomForestModel,
        GaussianNBModel
    ]

    best_model, average_score = train_models(X_train, X_test, y_train, y_test, models)
    
    print(f"Best model: {best_model.__class__.__name__} with average accuracy: {average_score}")
    best_model.train(X_train, y_train)
    y_pred = best_model.predict(X_test)
    plot_results(y_test, y_pred)


if __name__ == "__main__":
    main()