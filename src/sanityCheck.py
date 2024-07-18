from utils import *
from config import DATA_FILES, FAISS_INDEX_PATH
import pandas as pd
import os


def metrics_for_basic_queries():
    queries = [
        "What is VeriSoft framework?",
        "How to use the @DriverCapabilities annotation?",
        "What is the difference between the local thread and the global store?",
        "What is the EnvConfig class and how to use it?",
        "Show an example of how to perform a retry in the verisoft framework",
        "What is the object repository and how to use it?",
        "Show an example of how to add a value in the application.properties file and use it in a test / class",
        "I created Propertis file for using in tests. how can I read specific property from My Properties file",
        "Is there infrastructure support for reporting to XRAY? And if so, how do you use it?",
        "What is soft assert and how to use it in my test?"
    ]

    # Define search grid configurations
    search_grid = [
                      {"search_kwargs": {"k": k}} for k in [5, 10, 15]
                  ] + [
                      {"search_type": "mmr", "search_kwargs": {"k": k, "lambda_mult": lm}} for k in [5, 10] for lm in
                      [0.25, 0.5, 0.75]
                  ] + [
                      {"search_type": "similarity_score_threshold", "search_kwargs": {"score_threshold": st}} for st in
                      [0.5, 0.7, 0.8, 0.9]
                  ] + [
                      {"search_type": "mmr", "search_kwargs": {"k": k, "fetch_k": fk}} for k in [5, 10] for fk in
                      [30, 50, 70]
                  ] + [
                      {"search_kwargs": {"filter": {"paper_title": title}}} for title in ["GPT-4 Technical Report"]
                  ]

    metrics_df = pd.DataFrame(columns=["Query", "Search Option", "Max Score", "Mean Score", "Min Score"])

    if not os.path.exists(FAISS_INDEX_PATH):
        loader = load_files(folder_path=DATA_FILES)
        documents = split_text(loader=loader)
        embedding_model = load_embedding_model(device="cpu")
        vectorstore = FAISS.from_documents(documents=documents, embedding_model=embedding_model)
        save_embeddings(vectorstore, FAISS_INDEX_PATH)
    else:
        embedding_model = load_embedding_model(device="cpu")
        vectorstore = load_embeddings(FAISS_INDEX_PATH, embedding_model)

    for user_query in queries:
        for option in search_grid:
            print(user_query, option)
            query = ("I am an automation developer using the VeriSoft framework to write my tests. "
                     "I would like you to base your answer on the documentation and the code examples provided here. "
                     "My question is: ") + user_query

            retriever = vectorstore.as_retriever(**option)
            retrieved_documents = retriever.get_relevant_documents(query)

            if not retrieved_documents:
                print(f"No documents retrieved for query '{user_query}' with option {option}")
                continue
            reranker_model = load_reranker_model(device="cpu")
            sorted_docs_and_scores = rerank_docs(reranker_model, query, retrieved_documents)
            reranked_docs, scores = zip(*sorted_docs_and_scores)

            max_score = max(scores)
            mean_score = sum(scores) / len(scores)
            min_score = min(scores)

            temp_df = pd.DataFrame({
                "Query": [user_query],
                "Search Option": [str(option)],
                "Num of Documents":[len(scores)],
                "Max Score": [max_score],
                "Mean Score": [mean_score],
                "Min Score": [min_score]
            })
            metrics_df = pd.concat([metrics_df, temp_df], ignore_index=True)
            print(metrics_df)

    return metrics_df


def analyze_best_config(metrics_df):
    grouped_metrics = metrics_df.groupby("Search Option").agg({
        "Max Score": "mean",
        "Mean Score": "mean",
        "Min Score": "mean",
        "Num of Documents": "mean"
    }).reset_index()

    sorted_metrics = grouped_metrics.sort_values(by="Mean Score", ascending=False)
    return sorted_metrics


if __name__ == '__main__':
   # metrics_df = metrics_for_basic_queries()
    #metrics_df.to_csv("metrics2.csv")
    metrics_df=pd.read_csv('metrics2.csv')
    best_configs = analyze_best_config(metrics_df)
    best_configs.to_csv("best_configs3.csv")
    print(best_configs)
