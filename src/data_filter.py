# List of programming languages to filter by
programming_languages = ['python', 'java', 'javascript', 'c#', 'php', 'c++', 'r', 'swift', 'objective-c', 'kotlin',
                         'php-7.2', 'php-8', 'ruby-on-rails', 'ruby', 'c', 'go', 'scala', 'rust', 'dart', 'elixir']


# Function to check if any programming language is in the tags
def contains_programming_language(tags):
    return any(lang in tags for lang in programming_languages)


def filter_dataframe_on_languages(df):
    # Filter the DataFrame
    filtered_df = df[df['tags'].apply(contains_programming_language)]
    return filtered_df

def filter_dataframe_on_size(df, min_size, max_size):
    # Group by 'label' and filter out groups with fewer than 'size' entries
    filtered_df = df.groupby('label').filter(lambda x: len(x) >= min_size)

    # Keep only the top 'size' entries for each label
    filtered_df = filtered_df.groupby('label').head(max_size)

    return filtered_df