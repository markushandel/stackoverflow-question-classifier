def classify_datapoints(df, y_column_name):
    _, bucket_conditions = create_buckets(df, y_column_name)

    def classify(row):
        for bucket_num, condition in bucket_conditions.items():
            if condition.loc[row.name]:
                return bucket_num
        return None  # In case a value doesn't fit any bucket

    df['bucket'] = df.apply(classify, axis=1)
    return df
