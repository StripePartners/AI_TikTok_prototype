# import streamlit as st
# import pandas as pd
# import ast
# import ollama
# import os
# import sys

# from openai import OpenAI
# from nltk import sent_tokenize


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




# warren_logo_path = "V2 young warren logo.png"
# st.set_page_config(page_title="Young Warren",page_icon=warren_logo_path)
# st.image(warren_logo_path,width = 100)
# st.title("pov: ur tired of fake finance bros")
# st.write("young warren is ready to be in your corner")

# short_col, long_col = st.columns([0.33,0.67])
# alarm = 0 # check if a video was chosen yet
# with short_col:
#     try:
#         st.subheader("Video")
#         st.video(os.path.join("Finance_TikTok_prototype/corpus/videos","video"+str(st.session_state["user_select_video"]["index"]) + ".mp4"))
#         alarm = 1
#     except:
#         st.write("You need to pick a video to analyse first.")


# with long_col:
#     st.subheader("Bot Dialogue")
#     if alarm == 1:
#         chatbot("What is up?")
#     else:
#         chatbot("Waiting for a video to analyse")
    
# # I believe this goes in the file where all the functionality is configured, at the end
# st.markdown("""
# <style>
#   .stChatInput, .stChatMessage, .stChatMessageAvatarUser, .stExpander, button, .stDataFrameResizable, table, .stCheckbox span, .stWidgetLabel div, .stNumberInputContainer div, .stExpander details, .stDialog div {
#   	border-radius: none !important;
#   }
# </style>
# """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import ast
import ollama
import os
import sys

from openai import OpenAI
from nltk import sent_tokenize


def chatbot(start_msg):
    # Set up the chat bot
  
    #Initialize conversation history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Initialize models
    if "model" not in st.session_state:
        st.session_state["model"] = "llama3:8b"

    # Display chat messages from history on app rerun
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(start_msg):
        # add latest message to history in format {role, content}
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=warren_logo_path):
            message = st.write_stream(model_res_generator())
            st.session_state["messages"].append({"role": "assistant", "content": message})


def model_res_generator():
    stream = ollama.chat(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]


warren_logo_path = "V2 young warren logo.png"
st.set_page_config(page_title="Young Warren", page_icon=warren_logo_path)
st.image(warren_logo_path, width=100)
st.title("pov: ur tired of fake finance bros")
st.write("young warren is ready to be in your corner")

short_col, long_col = st.columns([0.33, 0.67])
alarm = 0  # check if a video was chosen yet
with short_col:
    try:
        st.subheader("Video")
        st.video(os.path.join("Finance_TikTok_prototype/corpus/videos", "video" + str(st.session_state["user_select_video"]["index"]) + ".mp4"))
        alarm = 1
    except:
        st.write("You need to pick a video to analyse first.")

with long_col:
    st.subheader("Bot Dialogue")
    if alarm == 1:
        chatbot("What is up?")
    else:
        chatbot("Waiting for a video to analyse")

# Custom CSS to make the chat input box stationary at the bottom
st.markdown("""
<style>
  .stChatInput {
    position: fixed;
    bottom: 0;
    left: 15%; /* Adjust based on the sidebar width */
    width: 70%;
    z-index: 1000;
  }
  .stChatMessage {
    overflow-y: auto;
    max-height: calc(100vh - 150px); /* Adjust based on your header and other content height */
  }
  .stChatMessageContainer {
    display: flex;
    flex-direction: column-reverse;
  }
</style>
""", unsafe_allow_html=True)