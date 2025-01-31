import streamlit as st
import pandas as pd
import ast
import ollama

from openai import OpenAI
from nltk import sent_tokenize


# Set up custom SP colours
color1 = '#FB5200' #tangerine
color2 = '#FFB7DE' #flamingo
color3 = '#8886FF' #violet
color4 = '#00DB90' #matcha
color5 = '#007B94' #teal
color6 = '#FFC736' #sunflower



def highlight_function(
                        extract, # the text extract to check if it belongs to subset of extracts
                        selected_extracts, # the subset of extracts predetermined to belong to one theme
                        color # the color of the theme
                        ):
    
    if extract.lower() in selected_extracts or extract in selected_extracts:
        return f''' <span style ="background-color:{color}"> {extract} </span> '''
    else:
        return extract


def model_res_generator():
    stream = ollama.chat(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]

def video_analysis():
    st.subheader("Check out a selection of videos")
    df = pd.read_csv("https://docs.google.com/spreadsheets/d/1naC0k4dQUOXXWEmSdLR3EVbyr8mBUYZ2KwZziwSleUA/export?gid=126339389&format=csv")

    #st.dataframe(df)
    #st.scatter_chart(df,x = "Index",y="View Count",color ="Investment Category")


    # Present a selection of videos to choose from for further analysis
    investment_categories = set(list(df["Investment Category"]))
    selected_category = st.selectbox("Select the investment category you want to learn more about!",investment_categories)

    videos_to_analyse = df["Video Title"][df["Investment Category"]==selected_category].head(5)
    indexes_to_analyse = list(df["Index"][df["Investment Category"]==selected_category].head(5))

    selected_video = st.selectbox("Choose video to analyse",videos_to_analyse)
    index = [indexes_to_analyse[i] for i,x in enumerate(videos_to_analyse) if x == selected_video][0]

    st.video("Finance_TikTok_prototype/corpus/videos/video1.mp4")

    # Analyse selected video
    st.markdown(f'''Check the prevalence of the following themes in the video: \\
            <span style ="background-color:{color1}"> Authority Bias </span> \\
            <span style ="background-color:{color2}"> Fear of Missing out (FOMO) </span> \\
            <span style ="background-color:{color3}"> Over confidence </span> \\
            <span style ="background-color:{color4}"> Loss Aversion </span> \
            ''',unsafe_allow_html=True
            )


    # Pre-identified extracts falling under each theme
    authority_transcript_extracts = ast.literal_eval(df["transcript_authority_bias_extracts"][index])
    authority_caption_extracts = ast.literal_eval(df["ocr_authority_bias_extracts"][index])

    fomo_transcript_extracts = ast.literal_eval(df["transcript_fomo_extracts"][index])
    fomo_caption_extracts = ast.literal_eval(df["ocr_fomo_extracts"][index])

    confidence_transcript_extracts = ast.literal_eval(df["transcript_confidence_boosting_extracts"][index])
    confidence_caption_extracts = ast.literal_eval(df["ocr_confidence_boosting_extracts"][index])
    
    loss_transcript_extracts = ast.literal_eval(df["transcript_loss_aversion_extracts"][index])
    loss_caption_extracts = ast.literal_eval(df["ocr_loss_aversion_extracts"][index])


    # Print coded transcript

    st.subheader("Transcript of selected video")
    transcript = df["transcript"][index]

    coded_transcript = ""
    for sentence in sent_tokenize(transcript):
        sentence = highlight_function(sentence, authority_transcript_extracts, color1)
        sentence = highlight_function(sentence, confidence_transcript_extracts, color3)
        sentence = highlight_function(sentence, loss_transcript_extracts, color4)
        sentence = highlight_function(sentence, fomo_transcript_extracts, color2)

        coded_transcript += sentence

    st.markdown(coded_transcript,unsafe_allow_html=True)


    # Print coded ocr captions
    st.subheader("Captions used in video")
    ocr_captions = ast.literal_eval(df["OCR_captions"][index])

    coded_captions = []
    for caption in ocr_captions:
        caption = highlight_function(caption, authority_caption_extracts, color1)
        caption = highlight_function(caption, fomo_caption_extracts, color2)
        caption = highlight_function(caption, confidence_caption_extracts, color3)
        caption = highlight_function(caption, loss_caption_extracts, color4)

        coded_captions.append(caption)

    st.markdown(', '.join(coded_captions),unsafe_allow_html=True)

def chatbot():
    # Set up the chat bot
    st.subheader("Find out more with Grandpa Warren")

    #Initialize conversation history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Initialize models
    if "model" not in st.session_state:
        st.session_state["model"] = "llama3:8b"

    models = [model.model for model in ollama.list()["models"]]
    print(models)
    st.session_state["model"] = st.selectbox("Choose your model", models)

    # Display chat messages from history on app rerun
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        # add latest message to history in format {role, content}
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message = st.write_stream(model_res_generator())
            st.session_state["messages"].append({"role": "assistant", "content": message})



#______The structure of the prototype______'

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    

# Set up header of app

st.title("👴💵📖 Grandpa Warren")
st.caption("pov: ur tired of fake finance bros")







