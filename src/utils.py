import os
import re
from typing import List
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import Language
from sentence_transformers import CrossEncoder
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs


def remove_headers_footers(text: str) -> str:
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        if not re.match(r'^(Page \d+|Footer|Header)', line.strip(), re.IGNORECASE):
            clean_lines.append(line)
    return '\n'.join(clean_lines)


def fix_broken_sentences(text: str) -> str:
    lines = text.split('\n')
    fixed_text = ""
    for line in lines:
        if line.endswith('-'):
            fixed_text += line[:-1]
        else:
            fixed_text += line + ' '
    return fixed_text.strip()


def remove_expand_phrases(text: str) -> str:
    """Remove lines containing 'Click here to expand...'."""
    return re.sub(r'Click here to expand\.\.\.', '', text, flags=re.MULTILINE)


def remove_apache_license(doc: str) -> str:
    """Post-processor to remove the Apache License comment block from the document content."""
    # Define the Apache License comment block pattern
    apache_license_pattern = re.compile(
        r"/\*.*?Licensed under the Apache License, Version 2\.0.*?\*/",
        re.DOTALL)

    # Remove the Apache License block from the document content
    doc = re.sub(apache_license_pattern, "", doc)

    # Optionally, you can strip any leading/trailing whitespace
    doc = doc.strip()

    return doc


def remove_metadata(text: str) -> str:
    lines = text.split('\n')
    clean_lines = [line for line in lines if not re.match(r'^<meta', line.strip(), re.IGNORECASE)]
    return '\n'.join(clean_lines)


def remove_line_indexes(text: str) -> str:
    """Remove lines that only contain numbers."""
    return re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)


def remove_indexes_in_code_blocks(text: str) -> str:
    """Remove indexes from code blocks."""
    return re.sub(r'^\d+\s', '', text, flags=re.MULTILINE)


def load_files(folder_path: str) -> List[UnstructuredFileLoader]:
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
    loaders = [UnstructuredFileLoader(
        file,
        post_processors=[clean_extra_whitespace, group_broken_paragraphs, remove_headers_footers,fix_broken_sentences,
                         remove_metadata, remove_expand_phrases, remove_indexes_in_code_blocks, remove_line_indexes], )
        for file in files]
    return loaders


def rerank_docs(reranker_model, query, retrieved_docs):
    query_and_docs = [(query, r.page_content) for r in retrieved_docs]
    scores = reranker_model.predict(query_and_docs)
    return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)


def rerank_docs_2_sources(reranker_model, query, retrieved_docs, retrieved_docs2):
    query_and_docs = [(query, r.page_content) for r in retrieved_docs + retrieved_docs2]
    scores = reranker_model.predict(query_and_docs)
    return sorted(list(zip(retrieved_docs + retrieved_docs2, scores)), key=lambda x: x[1], reverse=True)[:10]


def load_java_repository(repo_path):
    loader = GenericLoader.from_filesystem(
        repo_path + "/src",
        glob="**/*",
        suffixes=[".java", ".properties"],
        exclude=[],
        parser=LanguageParser("java", parser_threshold=700),

    )
    return loader


def split_and_process_repositories(loader):
    documents = loader.load()
    post_processors = [remove_apache_license, clean_extra_whitespace, group_broken_paragraphs]

    for i in range(len(documents)):
        content = documents[i].page_content
        for processor in post_processors:
            content = processor(content)
        documents[i].page_content = content
    java_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA, chunk_size=5000, chunk_overlap=200
    )
    texts = java_splitter.split_documents(documents)
    return texts


def load_text_files(folder_path: str = "data") -> List[UnstructuredFileLoader]:
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    loaders = [UnstructuredFileLoader(
        file,
        post_processors=[clean_extra_whitespace, group_broken_paragraphs, remove_headers_footers, fix_broken_sentences,
                         remove_metadata, remove_expand_phrases, remove_indexes_in_code_blocks,
                         remove_line_indexes, ], )
        for file in files]
    return loaders


def split_text(loaders: List[UnstructuredFileLoader], separators=None, chunk_size=1000):
    if separators is None:
        separators = ["\n\n\n", "\n\n"]
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,chunk_size=chunk_size,chunk_overlap=200,length_function=len, is_separator_regex=False)
    docs = []
    for loader in loaders:
        docs.extend(loader.load_and_split(text_splitter=text_splitter))
    return docs


def load_embedding_model(model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cpu") -> HuggingFaceBgeEmbeddings:
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs,
                                               encode_kwargs=encode_kwargs)
    return embedding_model


def load_reranker_model(reranker_model_name: str = "BAAI/bge-reranker-large", device: str = "cpu") -> CrossEncoder:
    reranker_model = CrossEncoder(model_name=reranker_model_name, max_length=512, device=device)
    return reranker_model


def save_embeddings(vectorstore, file_path: str):
    vectorstore.save_local(file_path)


def load_embeddings(file_path: str, embedding_model) -> FAISS:
    return FAISS.load_local(file_path, embedding_model, allow_dangerous_deserialization=True)


def create_compression_retriever(base_retriever, reranker_model) -> ContextualCompressionRetriever:
    embeddings_filter = EmbeddingsFilter(embeddings=reranker_model, similarity_threshold=0.5)
    compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                           base_retriever=base_retriever)
    return compression_retriever


def build_prompt(query, context):
    query_part = f"**Query**:\n{query}\n\n"
    context_part = "**Context**:\n"
    for idx, (doc, score) in enumerate(context):
        context_part += f"{idx + 1}. {doc.page_content.strip()}\n\n"
    # Instructions for the LLM
    instructions = ("**Instructions**: Using the context above, provide a detailed explanation"
                    " and example implementation to address the query.")

    return query_part + context_part + instructions
