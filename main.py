import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
import pickle
import time
import faiss  # Direct FAISS usage
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import cohere  # Cohere API
import os
os.environ["HF_HUB_OFFLINE"] = "1"

# Cohere API key
COHERE_API_KEY = "r9KRHcB0jl3ryx4AhfBvZhilggmHGlsCHv6zSjsS"  # Replace with your actual Cohere API key
co = cohere.Client(COHERE_API_KEY)  # Initialize Cohere client

# Load the SentenceTransformer model globally
model = SentenceTransformer('./local_model')
  # Hugging Face transformer model for embeddings

# Streamlit app interface with improved styling
st.set_page_config(
    page_title="Equity Research Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding: 20px;
        position: relative;
        animation: borderAnimation 5s infinite;
    }
    
    .sidebar-header {
        text-align: center;
        color: #1E88E5;
        padding: 15px;
        position: relative;
        animation: borderAnimation 5s infinite;
    }
    
    @keyframes borderAnimation {
        0% { box-shadow: 0 0 0 0 rgba(30, 136, 229, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(30, 136, 229, 0); }
        100% { box-shadow: 0 0 0 0 rgba(30, 136, 229, 0); }
    }
    .subheader {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
    }
    .source-link {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .stExpander {
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }
    .query-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App title with custom styling
st.markdown("<div class='main-header'>Equity Research Tool 📈</div>", unsafe_allow_html=True)

# Sidebar with improved styling
with st.sidebar:
    st.markdown("<div class='sidebar-header'><h2>News Article URLs</h2></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Get URLs from user input with better UI
    urls = []
    for i in range(3):
        url = st.text_input(f"URL {i+1}", key=f"url_{i}", 
                           help="Enter a news article URL to analyze")
        urls.append(url)
    
    process_url_clicked = st.button("Process URLs", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This tool analyzes news articles and answers your questions based on their content using AI.")

file_path = "faiss_store_hf.pkl"  # File to save the FAISS index

main_placeholder = st.empty()

# Process URLs when the button is clicked
if process_url_clicked:
    with st.spinner("Processing URLs... Please wait."):
        try:
            valid_urls = [url for url in urls if url]  # Remove empty URLs
            if not valid_urls:
                st.error("No valid URLs provided.")
                raise ValueError("No valid URLs provided.")

            # Initialize progress bar
            progress_bar = st.progress(0)
            total_urls = len(valid_urls)

            # Load data manually using requests and BeautifulSoup
            data = []
            for i, url in enumerate(valid_urls):
                response = requests.get(url)
                if response.status_code == 200:  # Changed from 500 to 200 for successful responses
                    soup = BeautifulSoup(response.content, "html.parser")
                    text = soup.get_text()
                    doc = Document(page_content=text, metadata={"source": url})
                    data.append(doc)
                else:
                    st.warning(f"Failed to retrieve URL: {url}, status code: {response.status_code}")

                # Update progress bar
                progress_bar.progress((i + 1) / total_urls)  # Corrected progress calculation

            if not data:
                st.error("No data loaded from the URLs.")
                raise ValueError("No data loaded from the URLs.")

            # Split the text data into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)

            if not docs:
                st.error("No documents generated after splitting the text.")
                raise ValueError("No documents generated after splitting the text.")

            # Create embeddings using Hugging Face model (Sentence-Transformers)
            embeddings = model.encode([doc.page_content for doc in docs])  # Generate embeddings

            # Initialize FAISS index
            dimension = embeddings.shape[1]  # Embedding dimension from model
            index = faiss.IndexFlatL2(dimension)  # L2 distance index
            index.add(embeddings)  # Add embeddings to FAISS index

            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            time.sleep(2)

            # Save the FAISS index and docs to a pickle file
            with open(file_path, "wb") as f:
                pickle.dump({"index": index, "docs": docs}, f)
            main_placeholder.text("FAISS Index saved successfully!")
            st.success("Processing completed successfully!")

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")

# Function to use Cohere for question-answering with relevance check
def cohere_answer_question(question, context):
    # First, check if the question is relevant to the context
    relevance_check = co.chat(
        model='command',
        message=f"Question: {question}\n\nContext: {context}\n\nIs this question answerable based on the provided context? Answer with only 'yes' or 'no'.",
        temperature=0.5  # Adjusted temperature for more leniency
    )
    
    # If the question is deemed not relevant
    if "no" in relevance_check.text.lower():
        return "I'm sorry, but the information you're asking about doesn't appear to be within the scope of the articles provided. Please ask a question related to the content of the articles."
    
    # If relevant, proceed with answering
    response = co.chat(
        model='command',
        message=f"Question: {question}\n\nContext: {context}\n\nAnswer the question based on the context provided.",
        temperature=0.7
    )
    return response.text

# Process user queries
# Process user queries - simplified to remove enhanced display bar
st.markdown("<h2 class='subheader'>Ask Questions About Your Articles</h2>", unsafe_allow_html=True)
query = st.text_input("Question:", placeholder="What would you like to know about these articles?")

if query:
    if os.path.exists(file_path):
        with st.spinner("Searching for relevant information and generating answer..."):
            with open(file_path, "rb") as f:
                saved_data = pickle.load(f)
                index = saved_data["index"]
                docs = saved_data["docs"]

                # Convert query into embedding using the same SentenceTransformer model
                query_embedding = model.encode([query])

                # Search FAISS index for the closest documents
                distances, indices = index.search(query_embedding, k=3)  # Retrieve top 3 results

                # Get the relevant documents based on the search
                relevant_docs = [docs[i] for i in indices[0]]

                if relevant_docs:
                    context = ' '.join([doc.page_content for doc in relevant_docs])
                else:
                    st.error("No relevant documents found.")
                    raise ValueError("No relevant documents found.")

                # Use Cohere to answer the query
                result = cohere_answer_question(query, context)

                # Create unique_sources set before using it
                unique_sources = set()
                for doc in relevant_docs:
                    unique_sources.add(doc.metadata["source"])

                # Display the result with darker background for better text visibility
                st.markdown("<h2 class='subheader'>Answer</h2>", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color: #1a237e; color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #3949ab;'>{result}</div>", unsafe_allow_html=True)
                
                # Display sources with matching dark styling
                st.markdown("<h3 class='subheader'>Sources:</h3>", unsafe_allow_html=True)
                for source in unique_sources:
                    st.markdown(f"<div style='background-color: #1a237e; color: white; padding: 10px; border-radius: 5px; margin: 5px 0;'><a href='{source}' target='_blank' style='color: #90caf9;'>{source}</a></div>", unsafe_allow_html=True)
                
                # Add the current query and result to session state for history
                if 'query_history' not in st.session_state:
                    st.session_state.query_history = []
                
                # Store the query, answer, and sources in history
                st.session_state.query_history.append({
                    "query": query,
                    "answer": result,
                    "sources": list(unique_sources)
                })

# Display query history at the bottom of the page
# Display query history with improved styling
st.markdown("<h2 class='subheader'>Query History</h2>", unsafe_allow_html=True)
if 'query_history' in st.session_state and st.session_state.query_history:
    for i, item in enumerate(reversed(st.session_state.query_history)):  # Show newest first
        with st.expander(f"Query {len(st.session_state.query_history)-i}: {item['query']}"):
            st.markdown("<h4 style='color: #1E88E5;'>Answer</h4>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color: #1a237e; padding: 10px; border-radius: 5px;'>{item['answer']}</div>", unsafe_allow_html=True)

else:
    st.info("No queries yet. Ask a question to get started!")
















