import streamlit as st
import pandas as pd
import ast
import ollama
import os
import sys
from nltk import sent_tokenize


def theme_analysis(video_index):
    
    video_index = video_index-1
    
    # Pre-identified extracts falling under each theme
    authority_transcript_extracts = ast.literal_eval(df["transcript_authority_bias_extracts"][video_index])
    authority_caption_extracts = ast.literal_eval(df["ocr_authority_bias_extracts"][video_index])

    fomo_transcript_extracts = ast.literal_eval(df["transcript_fomo_extracts"][video_index])
    fomo_caption_extracts = ast.literal_eval(df["ocr_fomo_extracts"][video_index])

    confidence_transcript_extracts = ast.literal_eval(df["transcript_confidence_boosting_extracts"][video_index])
    confidence_caption_extracts = ast.literal_eval(df["ocr_confidence_boosting_extracts"][video_index])
    
    loss_transcript_extracts = ast.literal_eval(df["transcript_loss_aversion_extracts"][video_index])
    loss_caption_extracts = ast.literal_eval(df["ocr_loss_aversion_extracts"][video_index])

    # Print coded transcript
    transcript = df["transcript"][video_index]

    coded_transcript = ""
    for sentence in sent_tokenize(transcript):
        sentence = highlight_function(sentence, authority_transcript_extracts, color1)
        sentence = highlight_function(sentence, confidence_transcript_extracts, color3)
        sentence = highlight_function(sentence, loss_transcript_extracts, color4)
        sentence = highlight_function(sentence, fomo_transcript_extracts, color2)

        coded_transcript += sentence

    st.markdown("*Transcript of video*")
    st.markdown(coded_transcript,unsafe_allow_html=True)

    # Print coded ocr captions
    ocr_captions = ast.literal_eval(df["OCR_captions"][video_index])

    coded_captions = []
    for caption in ocr_captions:
        caption = highlight_function(caption, authority_caption_extracts, color1)
        caption = highlight_function(caption, fomo_caption_extracts, color2)
        caption = highlight_function(caption, confidence_caption_extracts, color3)
        caption = highlight_function(caption, loss_caption_extracts, color4)

        coded_captions.append(caption)

    st.markdown("*OCR Captions at start of video*")
    st.markdown(', '.join(coded_captions),unsafe_allow_html=True)

def highlight_function(
                        extract, # the text extract to check if it belongs to subset of extracts
                        selected_extracts, # the subset of extracts predetermined to belong to one theme
                        color # the color of the theme
                        ):

    if extract.lower() in selected_extracts or extract in selected_extracts:
        return f''' <span style ="background-color:{color}"> {extract} </span> '''
    else:
        return extract



# Set up custom SP colours
color1 = '#FB5200' #tangerine
color2 = '#FFB7DE' #flamingo
color3 = '#8886FF' #violet
color4 = '#00DB90' #matcha
color5 = '#007B94' #teal
color6 = '#FFC736' #sunflower



#st.set_page_config(page_title="👴💵📖 Grandpa Warren", layout="wide")
asset_path = './assets/'
warren_logo_path = asset_path + "V2 young warren logo.png"
st.set_page_config(page_title="Young Warren",page_icon=warren_logo_path)
st.image(warren_logo_path,width = 100)
st.title("pov: ur tired of fake finance bros")
st.write("young warren is ready to be in your corner")

# df = pd.read_csv("https://docs.google.com/spreadsheets/d/1naC0k4dQUOXXWEmSdLR3EVbyr8mBUYZ2KwZziwSleUA/export?gid=126339389&format=csv") # the whole sample
df = pd.read_csv("https://docs.google.com/spreadsheets/d/1naC0k4dQUOXXWEmSdLR3EVbyr8mBUYZ2KwZziwSleUA/export?gid=1702026903&format=csv") # small sample of videos

#st.dataframe(df)
#st.scatter_chart(df,x = "Index",y="View Count",color ="Investment Category")


# Present a selection of videos to choose from for further analysis
investment_categories = set(list(df["Investment Category"]))
investment_categories = list(investment_categories)
lc_investment_categories = [x.lower() for x in investment_categories]
#st.write("bestie, what are we investing in today?")
selected_category = st.selectbox("bestie, what are we investing in today?",lc_investment_categories)
selected_category = investment_categories[lc_investment_categories.index(selected_category)]

videos_to_analyse = list(df["Video Title"][df["Investment Category"]==selected_category].head(3))
indexes_to_analyse = list(df["Index"][df["Investment Category"]==selected_category].head(3))

if 'user_select_video' not in st.session_state:
    st.session_state['user_select_video'] = {"index":0, "transcript":df["transcript"][0], "ocr_captions":ast.literal_eval(df["OCR_captions"][0])} #or whatever default
user_select_video = st.session_state['user_select_video']


col1, col2, col3 = st.columns(3, gap = "small")
video_data_path = "assets/video_data/videos"
with col1:
    #st.caption(videos_to_analyse[0])
    st.video(os.path.join(video_data_path,"video"+str(indexes_to_analyse[0]) + ".mp4"))
    var_click1 = st.button("vibe check this one",type="primary",key = "button1",use_container_width=True)

    if var_click1:
        st.session_state['user_select_video'] = {"index":indexes_to_analyse[0],
                                                 "transcript":df[df['Index'] == indexes_to_analyse[0]]["transcript"].iloc[0],
                                                 "ocr_captions":ast.literal_eval(df[df['Index'] == indexes_to_analyse[0]]["OCR_captions"].iloc[0]),
                                                 "video_type_in_app": df[df['Index'] == indexes_to_analyse[0]]["video_type_in_app"].iloc[0]} #or whatever default
        st.session_state["messages"] = []  # Reset chatbot history
        st.switch_page("pages/1_Conversation.py")

with col2:
    #st.caption(videos_to_analyse[1])
    st.video(os.path.join(video_data_path,"video"+str(indexes_to_analyse[1]) + ".mp4"))
    var_click2 = st.button("vibe check this one", type="primary",key = "button2",use_container_width=True)

    if var_click2:
        st.session_state['user_select_video'] = {"index":indexes_to_analyse[1],
                                                 "transcript":df[df['Index'] == indexes_to_analyse[1]]["transcript"].iloc[0],
                                                 "ocr_captions":ast.literal_eval(df[df['Index'] == indexes_to_analyse[1]]["OCR_captions"].iloc[0]),
                                                 "video_type_in_app": df[df['Index'] == indexes_to_analyse[1]]["video_type_in_app"].iloc[0]} #or whatever default
        st.session_state["messages"] = []  # Reset chatbot history
        st.switch_page("pages/1_Conversation.py")


with col3:
    #st.caption(videos_to_analyse[2])
    st.video(os.path.join(video_data_path,"video"+str(indexes_to_analyse[2]) + ".mp4"))
    var_click3 = st.button("vibe check this one", type="primary",key = "button3",use_container_width=True)

    if var_click3:
        st.session_state['user_select_video'] = {"index":indexes_to_analyse[2],
                                                 "transcript":df[df['Index'] == indexes_to_analyse[2]]["transcript"].iloc[0],
                                                 "ocr_captions":ast.literal_eval(df[df['Index'] == indexes_to_analyse[2]]["OCR_captions"].iloc[0]),
                                                 "video_type_in_app": df[df['Index'] == indexes_to_analyse[2]]["video_type_in_app"].iloc[0]} #or whatever default
        st.session_state["messages"] = []  # Reset chatbot history
        st.switch_page("pages/1_Conversation.py")


# I believe this goes in the file where all the functionality is configured, at the end
st.markdown("""
<style>
  .stChatInput, .stChatMessage, .stChatMessageAvatarUser, .stExpander, button, .stDataFrameResizable, table, .stCheckbox span, .stWidgetLabel div, .stNumberInputContainer div, .stExpander details, .stDialog div {
  	border-radius: none !important;
  }
</style>
""", unsafe_allow_html=True)

        