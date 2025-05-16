import streamlit as st
import pandas as pd
import ast
import ollama
import anthropic
import os
import sys
import pickle
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from streamlit_float import float_parent, float_init, float_css_helper
from openai import OpenAI
from nltk import sent_tokenize
import time

# Add root directory (where AI_TIKTOK_prototype lives) to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt_builder import get_prompt
from retriever import retrieve_context
asset_path = './assets/'
warren_logo_path = asset_path + "V2 young warren logo.png"
st.set_page_config(page_title="Young Warren",page_icon=warren_logo_path)

float_init(theme=True, include_unstable_primary=False)

##### Define chatbot function #####
def chatbot(start_msg):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "model" not in st.session_state:
        st.session_state["model"] = "claude-3-5-sonnet-20240620"

    # Create scrollable container for messages
    with st.container():
        with st.container():
            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    # Float input at the bottom
    with st.container():
        float_parent(css=float_css_helper(width="30%", bottom="2rem", transition=0))
        prompt = st.chat_input(start_msg)
    
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})

        current_v_transcript = st.session_state.get("user_select_video", {}).get("transcript", "No transcript available.")
        current_v_type = st.session_state.get("user_select_video", {}).get("video_type_in_app", "unknown")

        retrieved_letters, docs_letter = retrieve_context(prompt, document_type='letters')
        behavioural_science_docs, docs_books = retrieve_context(prompt, document_type='books')
        system_prompt = get_prompt(retrieved_letters, behavioural_science_docs, current_v_transcript, video_in_app_type=current_v_type)
        st.session_state["system_prompt"] = system_prompt

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message = st.write_stream(model_res_generator(system_prompt))
            st.session_state["messages"].append({"role": "assistant", "content": message})

##### Function to generate model response #####
def model_res_generator(system_prompt):
    # This is crucial. It constructs the full message chain to send to the model:
    # Starts with the system prompt (sets context, persona, and rules)
    # Then adds the entire user-assistant conversation so far
    # The system prompt only appears once, at the top of the message list
    # messages = [{"role": "system", "content": context}] + st.session_state["messages"]
    
    # Start with the system message
    # messages = [{"role": "system", "content": system_prompt}]
    messages = []
    
    # Append user-assistant history ONLY
    for msg in st.session_state["messages"]:
        if msg["role"] in ["user", "assistant"]:
            messages.append(msg)

    # DEBUG: Print full message chain
    print("\n--- MESSAGES SENT TO MODEL ---")
    print(json.dumps(messages, indent=2))  # Pretty print for easier reading

    # # Connects to ollama (local LLM runner), with the full message chain
    # stream = ollama.chat(
    #     model=st.session_state["model"],
    #     messages=messages,
    #     stream=True
    #     # options={"num_predict": 150} # Set maximum number of tokens to predict
    # )

    # # Streams response text chunk by chunk
    # for chunk in stream:
    #     yield chunk["message"]["content"]

    max_retries = 3
    retry_delay = 2  # seconds

    client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

    for attempt in range(max_retries):
        try:
            with client.messages.stream(
                model=st.session_state["model"],
                system=system_prompt,
                messages=messages,
                max_tokens=400
                ) as stream:
                for text in stream.text_stream:
                    yield text
            break
        except Exception as e:
            if "overloaded" in str(e):
                st.error(f"Anthropic API is overloaded. Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                
            else:
                st.error(f"An unexpected error occurred: {e}.")
                break
    else:
        st.error("Failed to connect to Anthropic API a response after multiple attempts. Please try again later.")
        yield "Error: Unable to get a response from the model."

##### app functions #####
st.image(warren_logo_path,width = 100)
st.title("pov: ur tired of fake finance bros")
st.write("young warren is ready to be in your corner")

short_col, long_col = st.columns([0.33,0.67])
alarm = 0 # check if a video was chosen yet
with short_col:
    try:
        st.subheader("Video")
        st.video(os.path.join("assets/video_data/videos","video"+str(st.session_state["user_select_video"]["index"]) + ".mp4"))
        
        # Retrieve and display the transcript
        transcript = st.session_state["user_select_video"].get("transcript", "No transcript available.")
        st.subheader("Transcript")
        st.text_area("Video Transcript", transcript, height=200)
        alarm = 1

        print(st.session_state["user_select_video"].get("video_type_in_app", "unknown"))
        print(st.session_state["user_select_video"].get("username", "unknown"))

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

