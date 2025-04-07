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
import time


# Load FAISS index and metadata
vector_dbs_path = '/Users/zoeliou/Documents/GitHub/AI_TikTok_prototype/vector_dbs/'
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

# Define function to retrieve context
def retrieve_context(query, document_type, k=3):
    """Retrieve relevant documents from FAISS and return text with creator info"""
    docs = db.similarity_search(query, filter={"document_type": document_type}, k=k)
    all_content = []
    with st.spinner("Running..."):
        for doc in docs:
            all_content.append(doc.page_content) # store all content retrieved in a list
        time.sleep(5)
    
    return "\n\n".join(all_content), docs

### Prompt
# define prompts for different video types
fomo_prompt = """
While responding to the user, you have to following the instructions below:\n
Repurposing FOMO (Fear of Missing Out)\n
    - Channel FOMO toward long-term thinking:\n
        - Create urgency around positive financial habits rather than specific volatile investments.\n
        - Use timelines and visualizations showing the "cost of delay" for retirement savings or debt repayment.\n
        - For example, "Don't miss out on the power of compound interest - waiting even 5 years to start investing could cost you thousands in future growth"
    - Redirect FOMO to financial literacy:
        - Generate excitement about learning opportunities rather than get-rich-quick schemes\n
        - For example, "The #1 advantage wealthy people have isn't secret investments - it's financial knowledge. Here's what you're missing if you don't understand these three concepts...
"""

overconfidence_prompt = """
While responding to the user, you have to following the instructions below:\n
Be ethical use of confidence\n
- Project confidence in proven principles:\n
    - Be extremely confident about well-established financial wisdom\n
    - Use strong, decisive language when discussing fundamentals that have stood the test of time\n
    - Maintain a confident, authoritative tone when countering misinformation\n
    - For example, "Dollar-cost averaging consistently outperforms market timing for 90% of retail investors"
- Confidence calibration\n
    - Be transparent about confidence levels\n
    - Express certainty proportional to evidence quality\n
    - For example, "I'm 95% confident about this advice because it's backed by decades of research" vs. "This is a newer approach with promising but limited data
"""

authority_bias_prompt = """
While responding to the user, you have to following the instructions below:\n
Be responsible authority leveraging\n
- Democratize expert knowledge\n
    - Translate complex insights from trusted authorities into actionable steps for beginners\n
    - Position the chatbot as a conduit to expert wisdom, not the ultimate authority itself\n
    - For example, "Here's what Warren Buffett does that you can actually replicate"
- Build a trust network\n
    - Cite multiple authorities when they agree on principles\n
    - Explain credentials in relatable terms: "This economist has correctly predicted 7 of the last 10 market shifts"\n
    - Compare and contrast different expert opinions when appropriate\n
    """
mix_prompt = """xyz"""

def get_prompt(context, current_v_transcript, video_in_app_type='unknown'):
    # define commmon prompt
    common_prompt = f"""
        You are an investment assistant with access to the following retrieved documents:\n{context}\n\n
        You also know that users have watched the following video and thisis the video transcript: \n{current_v_transcript}\n\n
        Based on this information, answer the user's question. \n
        """

    if video_in_app_type == 'fomo':
        system_prompt = common_prompt + fomo_prompt
    elif video_in_app_type == 'overconfidence':
        system_prompt = common_prompt + overconfidence_prompt
    elif video_in_app_type == 'authority_bias':
        system_prompt = common_prompt + authority_bias_prompt
    elif video_in_app_type == 'mix':
        system_prompt = common_prompt + mix_prompt
    else:
        system_prompt = common_prompt

    print(system_prompt) # check the prompt
    return system_prompt

# Define chatbot function
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

        current_v_transcript = st.session_state.get("user_select_video", {}).get("transcript", "No transcript available.")
        current_v_type = st.session_state.get("user_select_video", {}).get("video_type_in_app", "unknown")
        context, docs = retrieve_context(prompt)
        
        # create system prompt based on video type
        system_prompt = get_prompt(context, current_v_transcript, video_in_app_type=current_v_type)
        
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

asset_path = '/Users/zoeliou/Documents/GitHub/AI_TikTok_prototype/assets/'
warren_logo_path = asset_path + "V2 young warren logo.png"
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
        
        # Retrieve and display the transcript
        transcript = st.session_state["user_select_video"].get("transcript", "No transcript available.")
        st.subheader("Transcript")
        st.text_area("Video Transcript", transcript, height=200)
        alarm = 1

        print(st.session_state["user_select_video"].get("video_type_in_app", "unknown"))

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

