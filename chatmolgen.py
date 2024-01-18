import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(layout="wide")
st.title("Chatbot Assistant for Molecule Generation")

components.iframe("http://localhost:8000", height=600, scrolling=True)