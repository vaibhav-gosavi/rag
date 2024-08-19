import streamlit as st
import requests

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
                # Replace localhost with the actual deployed Flask app URL
                flask_url = "http://127.0.0.1:5000/process"
                response = requests.post(
                    flask_url,
                    json={'url': url, 'question': question}
                )
            if response.status_code == 200:
                answer = response.json().get('answer', 'No answer found.')
                
                # Display only the answer in a green-colored box
                st.markdown(
                    f"""
                    <div style="border: 2px solid #28a745; padding: 10px; border-radius: 5px; background-color: #d4edda; color: #155724;">
                        <strong>Answer:</strong> {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error('Error: Could not retrieve answer.')
        else:
            st.warning('Please enter both a URL and a question.')

    st.markdown("""
        ---
        Made with ❤️ using Streamlit & Flask.
    """)

if __name__ == '__main__':
    run_streamlit()
