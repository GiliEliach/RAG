import pandas as pd
if __name__=="__main__":
# Load the metrics from the CSV file
    file_path = 'best_configs3.csv'
    metrics_df = pd.read_csv(file_path)

    # Set your criteria
    min_documents = 3  # Minimum number of documents you want to retrieve

    # Filter configurations that meet the minimum number of documents
    filtered_df = metrics_df[metrics_df['Num of Documents'] >= min_documents]

    # Sort the filtered configurations by Mean Score in descending order
    sorted_df = filtered_df.sort_values(by='Mean Score', ascending=False)

    # Find the best configuration that also meets the desired number of documents

    # If no exact match, take the best one from the sorted list
    best_config = sorted_df.head(1)

    # Display the best configuration
    print("Best Configuration:")
    print(best_config)