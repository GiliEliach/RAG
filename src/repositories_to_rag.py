import os
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import Language
import openai

from src.config import FAISS_INDEX_PATH_REPOS
from src.utils import load_embedding_model, load_reranker_model, rerank_docs, build_prompt, save_embeddings, \
    load_embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
if __name__=="__main__":

    repo_path = "../bitbucket_repos"
    if not os.path.exists(repo_path):
        repo = Repo.clone_from("https://bitbucket.org/nir_gallner/verisoftframeworkexamples/src/master/", to_path=repo_path)

    loader = GenericLoader.from_filesystem(
        repo_path + "/src",
        glob="**/*",
        suffixes=[".java", ".properties"],
        exclude=[],
        parser=LanguageParser(language="java", parser_threshold=700)
    )

    # Load documents
    documents = loader.load()

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA, chunk_size=5000, chunk_overlap=200
    )
    texts = python_splitter.split_documents(documents)


    openai.api_key = "sk-proj-XoYQqW8LyvXJwOHVeuVGT3BlbkFJ93lMmd43DLIFcayMSN9j"

    question = "write to me a code exmaple to login test using verisoft framework with capabilities and url injection"
    if not os.path.exists(FAISS_INDEX_PATH_REPOS):
        embedding_model = load_embedding_model(device="cpu")
        vectorstore = FAISS.from_documents(documents=texts, embedding=embedding_model)
        save_embeddings(vectorstore, FAISS_INDEX_PATH_REPOS)
    else:
        embedding_model = load_embedding_model(device="cpu")
        vectorstore = load_embeddings(FAISS_INDEX_PATH_REPOS, embedding_model)



    retriever = vectorstore.as_retriever(search_kwargs={'k': 7})

    reranker_model = load_reranker_model(device="cpu")
    retrieved_documents = retriever.get_relevant_documents(question)
    context = rerank_docs(reranker_model, question, retrieved_documents)

    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",  # or use "davinci" for GPT-3 or specify another model
        prompt=build_prompt(question, context),
    )
    # Print the answer
    print(response.choices[0].text.strip())
