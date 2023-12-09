import json
import os
import pandas as pd


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_data_sets(folder_path):
    all_data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        data = load_data(file_path)
        all_data.extend(data)

    df = pd.DataFrame(all_data)

    label_column = 'closed_reason'
    # Assign a default label 'x' if the label column is not set or is NaN
    if label_column not in df.columns or df[label_column].isnull().any():
        df[label_column] = df[label_column].fillna('valid-question')
    
    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    return df
