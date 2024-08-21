from utils import *
from config import *
import pandas as pd


def main():
    user_query = input("Please enter your query: ")
    query = ("I am an automation developer using the VeriSoft framework to write my tests."
             "I would like you to base your answer on the documentation and the code examples provided here. "
             "my question is: ") + user_query
    embedding_model = load_embedding_model(device="cpu")
    reranker_model = load_reranker_model(device="cpu")

    if not os.path.exists(FAISS_INDEX_PATH):
        loader = load_files(folder_path=DATA_FILES)
        documents = split_text(loaders=loader)
        vectorstore_pdfs = FAISS.from_documents(documents=documents, embedding=embedding_model)
        save_embeddings(vectorstore_pdfs, FAISS_INDEX_PATH)
    else:
        embedding_model = load_embedding_model(device="cpu")
        vectorstore_pdfs = load_embeddings(FAISS_INDEX_PATH, embedding_model)

    if not os.path.exists(FAISS_INDEX_PATH_REPOS):
        loader_repos = load_java_repository(REPOS_FILES)
        documents_repos = split_and_process_repositories(loader_repos)
        vectorstore_repos = FAISS.from_documents(documents=documents_repos, embedding=embedding_model)
        save_embeddings(vectorstore_repos, FAISS_INDEX_PATH_REPOS)
    else:
        embedding_model = load_embedding_model(device="cpu")
        vectorstore_repos = load_embeddings(FAISS_INDEX_PATH_REPOS, embedding_model)

    retriever_pdfs = vectorstore_pdfs.as_retriever(search_kwargs={'search_type': 'similarity_score_threshold',
                                                                  'search_kwargs': {'score_threshold': 0.5}})
    retriever_repos = vectorstore_repos.as_retriever(search_kwargs={'search_type': 'similarity_score_threshold',
                                                                    'search_kwargs': {'score_threshold': 0.7}})

    retrieved_documents = retriever_pdfs.get_relevant_documents(query)
    retrieved_documents_repos = retriever_repos.get_relevant_documents(query)

    context = rerank_docs_2_sources(reranker_model, query, retrieved_documents, retrieved_documents_repos)
    return build_prompt(query, context)

if __name__ == '__main__':
    print(main())
