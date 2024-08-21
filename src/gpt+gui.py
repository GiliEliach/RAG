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


# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key ="sk-proj-XoYQqW8LyvXJwOHVeuVGT3BlbkFJ93lMmd43DLIFcayMSN9j"
conversation_history = []
client = OpenAI(
    api_key="sk-proj-XoYQqW8LyvXJwOHVeuVGT3BlbkFJ93lMmd43DLIFcayMSN9j"
)
def chat_with_gpt4(user_input):
    global conversation_history
    question=user_input
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