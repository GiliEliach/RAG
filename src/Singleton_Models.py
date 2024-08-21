import os
from typing import List, Optional

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs

from src.config import FAISS_INDEX_PATH, FAISS_INDEX_PATH_REPOS


class DocumentProcessor:
    def __init__(self, folder_path: str = "data"):
        self.folder_path = folder_path
        self.loaders = self.load_text_files()

    def load_text_files(self) -> List[UnstructuredFileLoader]:
        files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.txt')]
        loaders = [
            UnstructuredFileLoader(
                file,
                post_processors=[clean_extra_whitespace, group_broken_paragraphs],
            )
            for file in files
        ]
        return loaders

    def split_text(self, separators=None, chunk_size=1000):
        if separators is None:
            separators = ["\n\n\n", "\n\n"]
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        docs = []
        for loader in self.loaders:
            docs.extend(loader.load_and_split(text_splitter=text_splitter))
        return docs


class EmbeddingModels:
    _instance = None

    def __new__(cls, embedding_model_name: str = "BAAI/bge-large-en-v1.5",
                reranker_model_name: str = "BAAI/bge-reranker-large", device: str = "cpu"):
        if cls._instance is None:
            cls._instance = super(EmbeddingModels, cls).__new__(cls)
            cls._instance.embedding_model = cls._instance.load_embedding_model(embedding_model_name, device)
            cls._instance.reranker_model = cls._instance.load_reranker_model(reranker_model_name, device)
        return cls._instance


    def load_embedding_model(self, model_name: str, device: str) -> HuggingFaceBgeEmbeddings:
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": True}
        embedding_model = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs,
                                                   encode_kwargs=encode_kwargs)
        return embedding_model

    def load_reranker_model(self, reranker_model_name: str, device: str) -> CrossEncoder:
        reranker_model = CrossEncoder(model_name=reranker_model_name, max_length=512, device=device)
        return reranker_model


class DocumentRetriever:
    #_instance = None
    _instances={}
    # def __new__(cls, embedding_models: EmbeddingModels, index_path: str):
    #     if cls._instance is None:
    #         cls._instance = super(DocumentRetriever, cls).__new__(cls)
    #         cls._instance.initialize(embedding_models, index_path)
    #     return cls._instance
    def __new__(cls, embedding_models: EmbeddingModels, index_path: str):
        if index_path not in cls._instances:
            instance = super(DocumentRetriever, cls).__new__(cls)
            instance.initialize(embedding_models, index_path)
            cls._instances[index_path] = instance
        return cls._instances[index_path]

    def initialize(self, embedding_models: EmbeddingModels, index_path: str):
        self.embedding_model = embedding_models.embedding_model
        self.reranker_model = embedding_models.reranker_model
        self.index_path = index_path
        self.vectorstore = self.load_or_create_vectorstore()

    def load_or_create_vectorstore(self):
        if not os.path.exists(self.index_path):
            processor = DocumentProcessor()
            documents = processor.split_text()
            vectorstore = FAISS.from_documents(documents=documents, embedding=self.embedding_model)
            self.save_embeddings(vectorstore)
        else:
            vectorstore = self.load_embeddings()
        return vectorstore

    def save_embeddings(self, vectorstore):
        vectorstore.save_local(self.index_path)

    def load_embeddings(self) -> FAISS:
        return FAISS.load_local(self.index_path, self.embedding_model, allow_dangerous_deserialization=True)

    def rerank_docs(self, query, retrieved_docs):
        query_and_docs = [(query, r.page_content) for r in retrieved_docs]
        scores = self.reranker_model.predict(query_and_docs)
        return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)

    def retrieve_and_rerank(self, query: str, k: int = 10):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        retrieved_documents = retriever.get_relevant_documents(query)
        return self.rerank_docs(query, retrieved_documents)


def build_prompt(query, context):
    query_part = f"**Query**:\n{query}\n\n"
    context_part = "**Context**:\n"
    for idx, (doc, score) in enumerate(context):
        context_part += f"{idx + 1}. {doc.page_content.strip()}\n\n"
    instructions = "**Instructions**:\nUsing the context above, provide a detailed explanation and example implementation to address the query."
    prompt = query_part + context_part + instructions
    return prompt


def main(
         query: Optional[
             str] = 'Show an example of how to add a value in the application.properties file and use it in a test / class',
         index_path: str = FAISS_INDEX_PATH, repos_index_path=FAISS_INDEX_PATH_REPOS):
    query = "I am an automation developer using the VeriSoft framework to write my tests. I would like you to base your answer on the content and code examples provided here. my question is: " + query

    embedding_models = EmbeddingModels(device="cpu")
    docs_retriever = DocumentRetriever(embedding_models, index_path)
    code_retriever = DocumentRetriever(embedding_models, repos_index_path)
    docs_context = docs_retriever.retrieve_and_rerank(query)
    code_context = code_retriever.retrieve_and_rerank(query)

    return build_prompt(query, docs_context)


if __name__ == '__main__':
    print(main())
