import os
from typing import List, Optional
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import re
import gradio as gr
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import openai
from openai import OpenAI

from src.config import FAISS_INDEX_PATH


def rerank_docs(reranker_model, query, retrieved_docs):
    query_and_docs = [(query, r.page_content) for r in retrieved_docs]
    scores = reranker_model.predict(query_and_docs)
    return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)


def load_text_files(folder_path: str = "data") -> List[UnstructuredFileLoader]:
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    loaders = [
        UnstructuredFileLoader(
            file,
            post_processors=[clean_extra_whitespace, group_broken_paragraphs],
        )
        for file in files
    ]
    return loaders


def split_text(loaders: List[UnstructuredFileLoader], separators=None, chunk_size=1000):
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
    #intro = "I need help understanding the following in my Java tests.\n\n"

    query_part = f"**Query**:\n{query}\n\n"

    context_part = "**Context**:\n"
    for idx, (doc, score) in enumerate(context):
        context_part += f"{idx + 1}. {doc.page_content.strip()}\n\n"

    # Instructions for the LLM
    instructions = "**Instructions**:\nUsing the context above, provide a detailed explanation and example implementation to address the query."

    # Concatenating all parts
    prompt = query_part + context_part + instructions

    return prompt

def main(file: str = "/content/drive/MyDrive/rag data/confluence", query: Optional[str] = 'Show an example of how to add a value in the application.properties file and use it in a test / class', index_path: str = FAISS_INDEX_PATH):

    embedding_model = load_embedding_model(device="cpu")
    vectorstore = load_embeddings(index_path, embedding_model)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    reranker_model = load_reranker_model(device="cpu")
    retrieved_documents = retriever.get_relevant_documents(query)
    context = rerank_docs(reranker_model, query, retrieved_documents)
    print(context)
    return build_prompt(query, context)


# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key ="sk-proj-XoYQqW8LyvXJwOHVeuVGT3BlbkFJ93lMmd43DLIFcayMSN9j"
conversation_history = []
client = OpenAI(
    api_key="sk-proj-XoYQqW8LyvXJwOHVeuVGT3BlbkFJ93lMmd43DLIFcayMSN9j"
)
def chat_with_gpt4(user_input):
    global conversation_history
    question=main(query=user_input)
    print(question)
    conversation_history.append({"role": "user", "content": question})

    response  = client.chat.completions.create(
        model="gpt-4",
        messages=conversation_history
    )

    response_text = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": response_text})

    return response_text



def respond(user_input, history):
    global conversation_history
    conversation_history = [{"role": "user", "content": item[0]} for item in history] + \
                           [{"role": "assistant", "content": item[1]} for item in history if item[1]]
    response = chat_with_gpt4(user_input)
    history.append((user_input, response))
    return history, history,''
css ="""
#chatbot {
    background-color: #f0f8ff;
    border-radius: 10px;
    padding: 10px;
}
#msg {
    background-color: #e6f7ff;
    border: 2px solid #007bff;
    border-radius: 5px;
    color: #007bff;
}
#clear {
    background-color: #007bff;
    color: white;
    border: none;
    padding: 10px;
    border-radius: 5px;
}
"""
if __name__ == '__main__':

    theme=gr.themes.Soft()
    with gr.Blocks(css=css,theme=theme) as demo:
        chatbot = gr.Chatbot(elem_id="chatbot")
        msg = gr.Textbox(placeholder="Type a message...", elem_id="msg")
        clear = gr.Button("Clear", elem_id="clear")

        def clear_history():
            global conversation_history
            conversation_history = []
            return [],''

        msg.submit(respond, [msg, chatbot], [chatbot, chatbot, msg])
        clear.click(clear_history, None, [chatbot, msg])

    demo.launch()