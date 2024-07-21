
from flask import Flask, request, jsonify
from utils import *
from config import DATA_FILES, FAISS_INDEX_PATH
import os

app = Flask(__name__)


@app.route('/ask', methods=['POST'])
def ask():
   data = request.json
   if not data or 'prompt' not in data:
       return jsonify({'error': 'Bad Request', 'message': 'Missing prompt in request'}), 400

   user_query = data['prompt']
   query = ("I am an automation developer using the VeriSoft framework to write my tests."
            "I would like you to base your answer on the documentation and the code examples provided here. "
            "my question is: ") + user_query

   if not os.path.exists(FAISS_INDEX_PATH):
       loader = load_files(folder_path=DATA_FILES)
       documents = split_text(loaders=loader)
       embedding_model = load_embedding_model(device="cpu")
       vectorstore = FAISS.from_documents(documents=documents, embedding=embedding_model)
       save_embeddings(vectorstore, FAISS_INDEX_PATH)
   else:
       embedding_model = load_embedding_model(device="cpu")
       vectorstore = load_embeddings(FAISS_INDEX_PATH, embedding_model)

   retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
   reranker_model = load_reranker_model(device="cpu")
   retrieved_documents = retriever.get_relevant_documents(query)
   context = rerank_docs(reranker_model, query, retrieved_documents)
   response = build_prompt(query, context)

   return jsonify({'response': response})


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5001)
