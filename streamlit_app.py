import os
import streamlit as st
from dotenv import load_dotenv

import retrieve_from_llama2 as llama2


###Global variables:###
st.session_state["llm_app"] = llama2
st.session_state["llm_chain"] = llama2.build_chain()
MAX_HISTORY_LENGTH=6

###Initial UI configuration:###
st.set_page_config(page_title="Financial Bot", page_icon="🚀", layout="wide")


def render_app():
    # Reduce font sizes for input text boxes and button sizes
    custom_css = """
        <style>
            .stTextArea textarea {font-size: 13px;}
            div[data-baseweb="select"] > div {font-size: 13px !important;}
            button {
                height: 30px !important;
                width: 150px !important;
                padding-top: 10px !important;
                padding-bottom: 10px !important;
            }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Hide Streamlit menu and footer
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.subheader("Hello 👋 I'm your AI Finance Bot 😀")

    # Set up/Initialize Session State variables
    if "chat_dialogue" not in st.session_state:
        st.session_state["chat_dialogue"] = []
    if "llm" not in st.session_state:
        st.session_state["llm"] = llama2
        st.session_state["llm_chain"] = llama2.build_chain()

    def clear_history():
        st.session_state["chat_dialogue"] = []

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_dialogue:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if len(st.session_state.chat_dialogue) >= int(MAX_HISTORY_LENGTH):
        st.session_state.chat_dialogue = st.session_state.chat_dialogue[:MAX_HISTORY_LENGTH - 1]
        clear_history()

    if prompt := st.chat_input("Type your question here..."):
        # Add user message to chat history
        st.session_state.chat_dialogue.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            answer = ""
            llm_chain = st.session_state["llm_chain"]
            chain = st.session_state["llm"]
            try:
                output = llama2.run_chain(llm_chain, prompt)  # Directly run the chain function
                answer = output  # Assume output is a string
                print("ANSWER",answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                answer = "I'm sorry I'm not able to respond to your question 😔"
            # Add assistant response to chat history
            st.session_state.chat_dialogue.append({"role": "assistant", "content": answer})
            answer_placeholder.markdown(answer)

        col1, col2 = st.columns([10, 4])
        with col1:
            pass
        with col2:
            st.button("Clear History", use_container_width=True, on_click=clear_history)

render_app()