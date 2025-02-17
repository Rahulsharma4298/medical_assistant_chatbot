import streamlit as st
from graph import chat


st.set_page_config(page_title="Medical Assistant Chatbot")
st.title("🧑‍⚕️ :blue[Medical Assistant]")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    with st.spinner('Processing ..'):
        assistant_response = chat(user_input)
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})