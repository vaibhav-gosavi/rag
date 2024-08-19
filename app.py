import faiss
import numpy as np
from flask import Flask, request, jsonify
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os
import nltk
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

# Preload models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm_pipeline = pipeline("text-generation", model="gpt2")

# Flask route for processing the URL and question
@app.route('/process', methods=['POST'])
def process_url():
    data = request.json
    url = data['url']
    question = data['question']

    # Step 1: Load Data from URL
    loader = UnstructuredURLLoader(urls=[url])
    documents = loader.load()

    # Step 2: Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=200,
        chunk_overlap=0
    )
    chunks = text_splitter.split_documents(documents)

    # Step 3: Create embeddings for each chunk
    embeddings = embedding_model.encode([chunk.page_content for chunk in chunks])

    # Step 4: Create a FAISS index for vector similarity search
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Step 5: Query the system
    def query_rag_system(query):
        query_embedding = embedding_model.encode([query])
        _, indices = index.search(query_embedding, k=6)
        retrieved_chunks = [chunks[i].page_content for i in indices[0]]

        context = " ".join(retrieved_chunks)
        prompt = (
            f"Context: {context}\n\n"
            f"Question: {query}\n"
            "Please provide a clear and concise answer based only on the context above.\n"
            f"Answer:"
        )

        response = llm_pipeline(
            prompt,
            max_new_tokens=50,
            truncation=True,
            temperature=0.2
        )
        return response[0]['generated_text']

    answer = query_rag_system(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)