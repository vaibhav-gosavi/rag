import faiss
import numpy as np
from flask import Flask, request, jsonify
import streamlit as st
# from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import requests

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
        _, indices = index.search(query_embedding, k=5)
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

# Streamlit UI
def run_streamlit():
    st.set_page_config(page_title="URL Question Answering System", layout="wide")
    
    st.title('📄 URL Question Answering System')
    st.markdown("""
        Enter a URL and ask a question. The system will retrieve relevant information from the page and provide a concise answer.
    """)

    url = st.text_input('Enter the URL:', '')
    question = st.text_input('Enter your question:', '')

    if st.button('Get Answer'):
        if url and question:
            with st.spinner('Processing...'):
                response = requests.post(
                    'http://127.0.0.1:5000/process',
                    json={'url': url, 'question': question}
                )
            if response.status_code == 200:
                answer = response.json().get('answer', 'No answer found.')
                st.success(f'**Answer:** {answer}')
            else:
                st.error('Error: Could not retrieve answer.')
        else:
            st.warning('Please enter both a URL and a question.')

    st.markdown("""
        ---
        Made with ❤️ using Streamlit & Flask.
    """)

if __name__ == '__main__':
    from threading import Thread

    # Run Flask in a separate thread
    def run_flask():
        app.run(debug=False, use_reloader=False)  # Disable Flask reloader for compatibility with Streamlit

    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    # Run Streamlit
    run_streamlit()
