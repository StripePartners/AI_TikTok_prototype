import streamlit as st
import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

##### Load FAISS index and metadata #####
vector_dbs_path = './vector_dbs/'
faiss_path = vector_dbs_path + "faiss_index"
metadata_path = faiss_path + "/faiss_metadata.pkl"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)

# Load metadata
with open(metadata_path, "rb") as f:
    metadata_list = pickle.load(f)
print(f"Loaded {len(metadata_list)} metadata entries")

##### TEST #####
# query = "What is the best way to invest in stocks?"
# results = db.similarity_search(query, k=3, filter={"document_type": "letters"})
# for res, score in results:
#     print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

##### Define function to retrieve context #####
def retrieve_context(query, document_type=None, k=3):
    """Retrieve relevant documents from FAISS and return text with creator info"""
    if document_type is not None:
        docs = db.similarity_search(query, filter={"document_type": document_type}, k=k) # specify the document types: letters or books
    else:
        docs = db.similarity_search(query, k=k)

    all_content = []
    with st.spinner("Running..."):
        for doc in docs:
            all_content.append(doc.page_content) # store all content retrieved in a list
        time.sleep(5)
    
    return "\n\n".join(all_content), docs