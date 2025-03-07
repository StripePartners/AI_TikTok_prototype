import streamlit as st
import pandas as pd
import ast
import ollama
import os
import sys
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from openai import OpenAI
from nltk import sent_tokenize


# Load FAISS index and metadata
vector_dbs_path = '/Users/zoeliou/Documents/GitHub/AI_TikTok_prototype/vector_dbs/'
faiss_path = vector_dbs_path + "faiss_index"
metadata_path = faiss_path + "/faiss_metadata.pkl"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
with open(metadata_path, "rb") as f:
    metadata_list = pickle.load(f)
print(f"Loaded {len(metadata_list)} metadata entries")

def retrieve_context(query, k=3):
    """Retrieve relevant documents from FAISS and return text with creator info"""
    docs = db.similarity_search(query, k=k)
    retrieved_info = []
    
    for doc in docs:
        creator = next((meta["creator"] for meta in metadata_list if meta["text"] == doc.page_content), "Unknown")
        retrieved_info.append(f"Creator: {creator}\nContent: {doc.page_content}")
    
    return "\n\n".join(retrieved_info)

def chatbot(start_msg):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "model" not in st.session_state:
        st.session_state["model"] = "llama3:8b"

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(start_msg):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        
        context = retrieve_context(prompt)
        system_prompt = f"""You are an AI assistant with the following knowledge base:\n{context}\n\nAnswer the user's question based on this information."""
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message = st.write_stream(model_res_generator(system_prompt))
            st.session_state["messages"].append({"role": "assistant", "content": message})

# def chatbot(start_msg):
#     # Set up the chat bot
  
#     #Initialize conversation history
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []

#     # Initialize models
#     if "model" not in st.session_state:
#         st.session_state["model"] = "llama3:8b"

#     # Display chat messages from history on app rerun
#     for message in st.session_state["messages"]:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     if prompt := st.chat_input(start_msg):
#         # add latest message to history in format {role, content}
#         st.session_state["messages"].append({"role": "user", "content": prompt})

#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant",avatar = warren_logo_path):
#             message = st.write_stream(model_res_generator())
#             st.session_state["messages"].append({"role": "assistant", "content": message})

# def model_res_generator():
#     stream = ollama.chat(
#         model=st.session_state["model"],
#         messages=st.session_state["messages"],
#         stream=True,
#     )
#     for chunk in stream:
#         yield chunk["message"]["content"]

def model_res_generator(context):
    messages = [{"role": "system", "content": context}] + st.session_state["messages"]
    stream = ollama.chat(
        model=st.session_state["model"],
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]


warren_logo_path = "assets/V2 young warren logo.png"
st.set_page_config(page_title="Young Warren",page_icon=warren_logo_path)
st.image(warren_logo_path,width = 100)
st.title("pov: ur tired of fake finance bros")
st.write("young warren is ready to be in your corner")

short_col, long_col = st.columns([0.33,0.67])
alarm = 0 # check if a video was chosen yet
with short_col:
    try:
        st.subheader("Video")
        st.video(os.path.join("video_data/videos","video"+str(st.session_state["user_select_video"]["index"]) + ".mp4"))
        alarm = 1
    except:
        st.write("You need to pick a video to analyse first.")


with long_col:
    st.subheader("Bot Dialogue")
    if alarm == 1:
        chatbot("What is up?")
    else:
        chatbot("Waiting for a video to analyse")
    
# I believe this goes in the file where all the functionality is configured, at the end
st.markdown("""
<style>
  .stChatInput, .stChatMessage, .stChatMessageAvatarUser, .stExpander, button, .stDataFrameResizable, table, .stCheckbox span, .stWidgetLabel div, .stNumberInputContainer div, .stExpander details, .stDialog div {
  	border-radius: none !important;
  }
</style>
""", unsafe_allow_html=True)

# st.markdown("""
# <style>
#   .stChatInput {
#     position: fixed;
#     bottom: 5;
#     right: 10;
#     width: 33%; /* Match the width of the long_col */
#     z-index: 1000;
#   }
#   .stChatMessageContainer {
#     display: flex;
#     flex-direction: column-reverse;
#     overflow-y: auto;
#     max-height: calc(100vh - 150px); /* Adjust based on your header and other content height */
#     padding-bottom: 60px; /* Adjust based on the height of the chat input box */
#   }
#   .stChatMessage {
#     overflow-y: auto;
#   }
# </style>
# """, unsafe_allow_html=True)

