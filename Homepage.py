
import streamlit as st
import pandas as pd
import ast
import anthropic
import os
import sys
import pickle
import json
import torch
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import re
#from streamlit_float import *

from openai import OpenAI
from nltk import sent_tokenize
import time
from streamlit.components.v1 import html

# Add root directory (where AI_TIKTOK_prototype lives) to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt_builder import get_prompt, get_prompt_consistency_eval
from retriever import retrieve_context

#float_init(theme=True, include_unstable_primary=False)


#torch.classes.__path__ = []

def set_selected_question(question_text):
    st.session_state["selected_question"] = question_text

def open_creator_profile(url):
    open_script= """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)

##### Define chatbot function #####
def chatbot(prompt):
    
    #button_css = float_css_helper(width="10rem", bottom="0rem", transition=0)
    #float_parent(css=button_css)

    if "messages" not in st.session_state: #  Initializes message history.
        st.session_state["messages"] = []

    if "model" not in st.session_state:
        st.session_state["model"] = "claude-3-5-sonnet-20240620" # Sets a default model if one hasn’t been chosen
    
    # Add LLM-generated start message
    # current_v_transcript = st.session_state.get("user_select_video", {}).get("transcript", "No transcript available.")
    # question_count = 3
    # start_prompt = f'''Based on the transcript {current_v_transcript}, briefly outline a short summary of the transcript (less than 20 words) followed by {question_count} short (10 words or less) specific questions relevant to information 
    #                 in the transcript that could start a conversation on financial advice. Begin the message with 'Hello! Nice to meet you!', followed by the summary introduced by the statement "The TikTok video discusses". 
    #                 After, write 'You could ask me things like:' followed by the questions.
    #                 Write this in markdown format such that the questions are written in white and highlighted in black and each question begins on a new line, 
    #                 leaving a new line in between questions. Questions should not be punctuated and should be writen in lower case.'''

    # model_response = model_res_non_generator(start_prompt)
    #st.write(model_response.content[0].text)

    model_response =  st.session_state["user_select_video"]["prompt"]
    message = st.chat_message("assistant",avatar=role_to_image["assistant"])
    intro = model_response.split("You could ask me things like:")[0]
    suggested_questions = re.findall(r'<span.*?>(.*?)</span>', model_response)
        
    with message:
        st.markdown(intro+"You could ask me things like:",unsafe_allow_html=True)
        for question in suggested_questions:
            st.markdown('<span id="button-prompt"></span>', unsafe_allow_html=True)
            st.button(question,
                  key=question,
                  on_click=set_selected_question,
                  args=(question,))
   
    #print(model_res_non_generator(start_prompt))
    
    #message.write(model_res_non_generator(start_prompt))
    #st.session_state["messages"].append({"role": "assistant", "content": message}) # "assistant": model response
    
    for message in st.session_state["messages"]: # Re-displaying the chat history
        with st.chat_message(message["role"],avatar = role_to_image[message["role"]]):
            st.markdown(message["content"])

    if "selected_question" in st.session_state and not prompt:
        prompt = st.session_state.pop("selected_question")
    
    if prompt:  # Waits for new user input
        
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

        with st.chat_message("user",avatar=role_to_image["user"]): # Renders user input immediately
            st.markdown(prompt)
        
        with st.chat_message("assistant",avatar=avatar_bot): # Streams and displays the assistant's response, then stores it in chat history
            message = st.write_stream(model_res_generator(system_prompt))
            st.session_state["messages"].append({"role": "assistant", "content": message}) # "assistant": model response





##### Function to generate non-stream model response #####
def model_res_non_generator(start_prompt):
    max_retries = 3
    retry_delay = 2  # seconds
    
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"]) #os.getenv("ANTHROPIC_API_KEY")  #st.secrets["ANTHROPIC_API_KEY"]

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

    client = anthropic.Client(api_key=st.secrets["ANTHROPIC_API_KEY"]) #os.getenv("ANTHROPIC_API_KEY") #st.secrets["ANTHROPIC_API_KEY"]

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
    else:
        st.session_state["order"] = 0

    i = indexes_to_analyse[st.session_state['order']]
    st.session_state["user_select_video"] = {"index":i,
                                        "transcript":df[df['Index'] == i]["transcript"].iloc[0],
                                        "ocr_captions":ast.literal_eval(df[df['Index'] == i]["OCR_captions"].iloc[0]),
                                        "video_type_in_app": df[df['Index'] == i]["video_type_in_app"].iloc[0],
                                        "creator_tag": df[df['Index'] == i]["creator_tag"].iloc[0],
                                        "creator_profile_url":df[df['Index'] == i]["creator_profile_url"].iloc[0],
                                        "prompt":df[df['Index'] == i]["prompt"].iloc[0] }
    st.session_state["messages"] = []  # Reset chatbot history



##### app functions #####
asset_path = './assets/'

avatar_bot = asset_path + "V4 avatar bot.png"
avatar_person = asset_path + "V4 avatar person.png"
role_to_image = {"assistant":avatar_bot,"user":avatar_person}

warren_logo_path = asset_path + "V4 COIN logo small.png"
st.set_page_config(page_title="Warren.ai",page_icon=warren_logo_path)
st.image(warren_logo_path,width = 100)
st.title("pov: ur tired of fake finance bros")
st.write("Warren.ai is ready to be in your corner")

short_col, long_col = st.columns([0.4,0.6])
alarm = 0 # check if a video was chosen yet


# Read dataset
df = pd.read_csv("https://docs.google.com/spreadsheets/d/1naC0k4dQUOXXWEmSdLR3EVbyr8mBUYZ2KwZziwSleUA/export?gid=1702026903&format=csv") # small sample of videos
indexes_to_analyse = list(df["Index"]) #(i for i in list(df["Index"]))
# print(df.columns)

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
                                        "creator_profile_url":df[df['Index'] == i]["creator_profile_url"].iloc[0],
                                        "prompt":df[df['Index'] == i]["prompt"].iloc[0]}


if "messages" not in st.session_state:      
    st.session_state["messages"] = []  # Reset chatbot history


with short_col:
           
    st.subheader("Watch this first")
    if st.session_state["order"]<6:

        st.video(os.path.join("assets/video_data/videos","video"+str(st.session_state["user_select_video"]["index"]) + ".mp4"))
        creator_tag = st.session_state["user_select_video"]["creator_tag"]
        creator_profile_url = st.session_state["user_select_video"]["creator_profile_url"]
        st.markdown('<span id="button-standard"></span>', unsafe_allow_html=True)
        st.button(creator_tag,
                  key=creator_tag,
                  on_click=open_creator_profile,
                  args=(creator_profile_url,),
                  use_container_width=True)
    else:
        st.write("")

with long_col:
    st.subheader("Then talk it out")
    #Initialise chat
    # prompt = st.chat_input("Type to chat")

    # with st.container(height = 432, border = None):  #manually set # of pixels for height of container

    #     chatbot(prompt)

    chat_container = st.container(height=487, border=None)
    with chat_container:
        chatbot(st.session_state.get("submitted_prompt", ""))

    with st.form(key="chat_form", clear_on_submit=True, enter_to_submit=True, border=False):
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            user_input = st.text_input(label="Type to chat",
                                       placeholder="Type to chat",
                                       label_visibility="collapsed",
                                       key="chat_input")
        with col2:
            submitted = st.form_submit_button("",icon=":material/send:", use_container_width=True)
        if submitted and user_input.strip():
            st.session_state["submitted_prompt"] = user_input
            del st.session_state["chat_input"]
            st.rerun()
        else:
            st.session_state["submitted_prompt"] = ""


# Choose a video to show next
st.markdown('<span id="button-standard"></span>', unsafe_allow_html=True)
var_click1 = st.button("Try a different video",key = "button1",use_container_width=True,on_click = callback, args = [indexes_to_analyse])


# I believe this goes in the file where all the functionality is configured, at the end
st.markdown("""
<style>
  .stChatInput, .stChatMessage, .stChatMessageAvatarUser, .stExpander, button, .stDataFrameResizable, table, .stCheckbox span, .stWidgetLabel div, .stNumberInputContainer div, .stExpander details, .stDialog div {
  	border-radius: none !important;
  }
  .element-container:has(style){
    display: none;
  }
  #button-prompt {
    display: none;
  }
  .element-container:has(#button-prompt) {
    display: none;
  }
  .element-container:has(#button-prompt) + div button {
    color:white;
    background-color:black;
    border: none;
    text-align: left;
    border-radius: 0;
  }
  #button-standard {
    display: none;
  }
  .element-container:has(#button-standard) {
    display: none;
  }
  .element-container:has(#button-standard) + div button {
    color:white;
    background-color:black;
    border: none;
    text-align: center;
    border-radius: 5px;
  }
  div[data-testid="InputInstructions"] > span:nth-child(1) {
    visibility: hidden;
  }
  .element-container:has(video) {
    margin-bottom: -7px !important;
  }
  .element-container:has(button[title="{creator_tag}"]) {
    margin-top: -7px !important;
  }
</style>
""", unsafe_allow_html=True)