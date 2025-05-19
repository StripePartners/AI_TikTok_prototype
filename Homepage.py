import streamlit as st
import pandas as pd
import ast
import ollama
import anthropic
import os
import sys
import pickle
import json
import torch
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from openai import OpenAI
from nltk import sent_tokenize
import time

# Add root directory (where AI_TIKTOK_prototype lives) to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt_builder import get_prompt, get_prompt_consistency_eval
from retriever import retrieve_context

#torch.classes.__path__ = [] 

##### Define chatbot function #####
def chatbot(start_msg):
    
    
    if "messages" not in st.session_state: #  Initializes message history.
        st.session_state["messages"] = []

    if "model" not in st.session_state:
        st.session_state["model"] = "claude-3-5-sonnet-20240620" # Sets a default model if one hasn’t been chosen

    # Add LLM-generated start message
    current_v_transcript = st.session_state.get("user_select_video", {}).get("transcript", "No transcript available.")
    bullet_points = 3
    start_prompt = f"Based on the transcript {current_v_transcript}, briefly outline {bullet_points} short (10 words or less) specific questions relevant to information in the transcript that could start a conversation on financial advice. Provide the questions in bullet format beginning with 'Hello! Nice to meet you! You could ask me things like:'"

    model_response = model_res_non_generator(start_prompt)
    #st.write(model_response.content[0].text)

    message = st.chat_message("assistant")
    message.write(model_response.content[0].text)
    #print(model_res_non_generator(start_prompt))
    
    #message.write(model_res_non_generator(start_prompt))
    #st.session_state["messages"].append({"role": "assistant", "content": message}) # "assistant": model response
    
    for message in st.session_state["messages"]: # Re-displaying the chat history
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(start_msg): # Waits for new user input
        st.session_state["messages"].append({"role": "user", "content": prompt}) # Adds the user message to history

        # Grabs the selected video transcript and its categorized behavior type
        current_v_transcript = st.session_state.get("user_select_video", {}).get("transcript", "No transcript available.")
        current_v_type = st.session_state.get("user_select_video", {}).get("video_type_in_app", "unknown")

        # Gets relevant info from both letters and behavioral science books
        retrieved_letters, docs_letter = retrieve_context(prompt, document_type='letters')
        behavioural_science_docs, docs_books = retrieve_context(prompt, document_type='books')
        
        # create system prompt based on video type
        # This is your initial context for the model — a custom “system prompt”
        system_prompt = get_prompt_consistency_eval(retrieved_letters, behavioural_science_docs, current_v_transcript)
        st.session_state["system_prompt"] = system_prompt

        with st.chat_message("user"): # Renders user input immediately
            st.markdown(prompt)
        
        with st.chat_message("assistant"): # Streams and displays the assistant's response, then stores it in chat history
            message = st.write_stream(model_res_generator(system_prompt))
            st.session_state["messages"].append({"role": "assistant", "content": message}) # "assistant": model response





##### Function to generate non-stream model response #####
def model_res_non_generator(start_prompt):
    max_retries = 3
    retry_delay = 2  # seconds
    
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) #os.getenv("ANTHROPIC_API_KEY")  #st.secrets[]

    for attempt in range(max_retries):
        try:
            text = client.messages.create(
                                            model=st.session_state["model"],
                                            max_tokens=400,
                                            messages=[{"role":"user","content":start_prompt}],
                                            ) 
            return text
        except Exception as e:
            if "overloaded" in str(e):
                st.error(f"Anthropic API is overloaded. Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                
            else:
                st.error(f"An unexpected error occurred: {e}.")
                break
    
    else:
        st.error("Failed to connect to Anthropic API a response after multiple attempts. Please try again later.")
        return "Error: Unable to get a response from the model."

    






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

    # Added sleep time to avoid API rate limits
    time.sleep(1.5)
    
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

    client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY")) #os.getenv("ANTHROPIC_API_KEY") #st.secrets[]

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


# callback to get to the next video (imitate generator functionality)
def callback(indexes_to_analyse):
    time.sleep(1.5)  # added sleep time to avoid rate limits
    print("Session state",st.session_state["order"],len((indexes_to_analyse)))

    if st.session_state["order"] < len(indexes_to_analyse) - 1: 
        st.session_state["order"] += 1
        
        i = indexes_to_analyse[st.session_state['order']]
        st.session_state["user_select_video"] = {"index":i,
                                            "transcript":df[df['Index'] == i]["transcript"].iloc[0],
                                            "ocr_captions":ast.literal_eval(df[df['Index'] == i]["OCR_captions"].iloc[0]),
                                            "video_type_in_app": df[df['Index'] == i]["video_type_in_app"].iloc[0],
                                            "creator_tag": df[df['Index'] == i]["creator_tag"].iloc[0],
                                            "creator_profile_url":df[df['Index'] == i]["creator_profile_url"].iloc[0] }
        st.session_state["messages"] = []  # Reset chatbot history
    else:
        st.write("Reached limit on videos to analyse.")


##### app functions #####
asset_path = './assets/'
warren_logo_path = asset_path + "V2 young warren logo.png"
st.set_page_config(page_title="Warren.ai",page_icon=warren_logo_path)
st.image(warren_logo_path,width = 100)
st.title("pov: ur tired of fake finance bros")
st.write("Warren.ai is ready to be in your corner")

short_col, long_col = st.columns([0.33,0.67])
alarm = 0 # check if a video was chosen yet


# Read dataset
df = pd.read_csv("https://docs.google.com/spreadsheets/d/1naC0k4dQUOXXWEmSdLR3EVbyr8mBUYZ2KwZziwSleUA/export?gid=1702026903&format=csv") # small sample of videos
indexes_to_analyse = list(df["Index"]) #(i for i in list(df["Index"]))
print(df.columns)

#Empty dictionary
if "order" not in st.session_state:
    st.session_state["order"] = 0
if "user_select_video" not in st.session_state:
    i = indexes_to_analyse[st.session_state['order']]
    st.session_state["user_select_video"] = {"index":i,
                                        "transcript":df[df['Index'] == i]["transcript"].iloc[0],
                                        "ocr_captions":ast.literal_eval(df[df['Index'] == i]["OCR_captions"].iloc[0]),
                                        "video_type_in_app": df[df['Index'] == i]["video_type_in_app"].iloc[0],
                                        "creator_tag": df[df['Index'] == i]["creator_tag"].iloc[0],
                                        "creator_profile_url":df[df['Index'] == i]["creator_profile_url"].iloc[0] }


if "messages" not in st.session_state:      
    st.session_state["messages"] = []  # Reset chatbot history


with short_col:
    
    #try:
        st.subheader("Step 1: Watch this")
        st.video(os.path.join("assets/video_data/videos","video"+str(st.session_state["user_select_video"]["index"]) + ".mp4"))
        #st.link_button(st.markdown(f''':blue[{st.session_state["user_select_video"]["creator_tag"]}]''',unsafe_allow_html=False),st.session_state["user_select_video"]["creator_profile_url"],type="tertiary")
        st.link_button(st.session_state["user_select_video"]["creator_tag"],st.session_state["user_select_video"]["creator_profile_url"],type="secondary")

        
        # Retrieve and display the transcript
        # transcript = st.session_state["user_select_video"].get("transcript", "No transcript available.")
        # st.subheader("Transcript")
        # st.text_area("Video Transcript", transcript, height=200)
        # alarm = 1

        # print(st.session_state["user_select_video"].get("video_type_in_app", "unknown"))

    #except:
    #    st.write("You need to pick a video to analyse first.")


with long_col:
    st.subheader("Step 2: Talk it out")
    with st.container(height = 435, border = None):  #manually set # of pixels for height of container
        if alarm == 1:
            chatbot("Type to chat")
        else:
            chatbot("Type to chat")
    

# Choose a video to show next
var_click1 = st.button("show me another one",type="secondary",key = "button1",use_container_width=True,on_click = callback, args = [indexes_to_analyse])


# I believe this goes in the file where all the functionality is configured, at the end
st.markdown("""
<style>
  .stChatInput, .stChatMessage, .stChatMessageAvatarUser, .stExpander, button, .stDataFrameResizable, table, .stCheckbox span, .stWidgetLabel div, .stNumberInputContainer div, .stExpander details, .stDialog div {
  	border-radius: none !important;
  }
</style>
""", unsafe_allow_html=True)

