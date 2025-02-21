import streamlit as st
import pandas as pd
import ast
import ollama
import os
import sys

from openai import OpenAI
from streamlit_float import *
from streamlit_extras.bottom_container import bottom
from nltk import sent_tokenize

float_init(theme=True, include_unstable_primary=False)

def chat_content():
    st.session_state['contents'].append(st.session_state.content)


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
        #avatar = None
        if message["role"] == "user":
            avatar = user_profile_path
        else:
            avatar = bot_profile_path    
        
        with st.chat_message(message["role"],avatar = avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input(start_msg):
        # add latest message to history in format {role, content}
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user",avatar = user_profile_path):
            st.markdown(prompt)

        with st.chat_message("assistant",avatar = bot_profile_path):
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




warren_logo_path = "Finance_TikTok_prototype/AI_TikTok_prototype/V2 young warren logo.png"
bot_profile_path = "Finance_TikTok_prototype/AI_TikTok_prototype/bot_profile.png"
user_profile_path = "Finance_TikTok_prototype/AI_TikTok_prototype/user_profile.png"
#st.set_page_config(page_title="Young Warren",page_icon=warren_logo_path)
st.image(warren_logo_path,width = 100)
st.title("pov: ur tired of fake finance bros")
st.write("young warren is ready to be in your corner")

short_col, long_col = st.columns([0.33,0.67])
alarm = 0 # check if a video was chosen yet
with short_col:
    
    var_click = st.button("<take me back", type="tertiary",key = "button",use_container_width=False)

    try:
        #st.subheader("Video")
        st.video(os.path.join("Finance_TikTok_prototype/corpus/videos","video"+str(st.session_state["user_select_video"]["index"]) + ".mp4"))
        alarm = 1
    except:
        st.write("You need to pick a video to analyse first.")
    
    if var_click == True:
        #st.session_state['user_select_video'] = {"index":indexes_to_analyse[2],"transcript":df["transcript"][indexes_to_analyse[2]],"ocr_captions":ast.literal_eval(df["OCR_captions"][indexes_to_analyse[2]])} #or whatever default
        st.switch_page("Homepage.py")


with long_col:
    #st.subheader("Bot Dialogue")
    st.markdown('###')
    with st.container():

        if alarm == 1:
            
            chatbot("What is up?")
        else:
            chatbot("Waiting for a video to analyse")
    
# I believe this goes in the file where all the functionality is configured, at the end
st.markdown("""
<style>
  * {
    border-radius: 0 !important;
  }
  .stChatInput, .stChatMessage, .stChatMessageAvatarUser, .stExpander, button, .stDataFrameResizable, table, .stCheckbox span, .stWidgetLabel div, .stNumberInputContainer div, .stExpander details, .stDialog div {
  	bordr-radius: 0 !important;
  }
  div.stButton > button:first-child{
  background-color:white;
  color:black;
  border-color:white;
  font-weight:bold;
  text-decoration:underline;
  }
  .stChatMessage:has([aria-label='Chat message from user']) {
    background: #F4F3EE;
    text-align: right;
  }
  [aria-label='Chat message from user'] {
    background-color: #F4F3EE;
  }
  .stSidebar {
    display: none;
  }
</style>
""", unsafe_allow_html=True)




