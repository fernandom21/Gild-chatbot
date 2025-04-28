import streamlit as st
from openai import OpenAI
import time
import re

placeholderstr = "Please input your command"
user_name = "Fernando"
user_image = "https://www.w3schools.com/howto/img_avatar.png"

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

def main():
    st.set_page_config(
        page_title='K-Assistant - The Residemy Agent',
        layout='wide',
        initial_sidebar_state='auto',
        menu_items={
            'Get Help': 'https://streamlit.io/',
            'Report a bug': 'https://github.com',
            'About': 'About your application: **Hello world**'
            },
        page_icon="img/favicon.ico"
    )

    # Show title and description.
    st.title(f"💬 {user_name}'s Chatbot")

    with st.sidebar:
        selected_lang = st.selectbox("Language", ["English", "繁體中文"], index=1)
        if 'lang_setting' in st.session_state:
            lang_setting = st.session_state['lang_setting']
        else:
            lang_setting = selected_lang
            st.session_state['lang_setting'] = lang_setting

        st_c_1 = st.container(border=True)
        with st_c_1:
            st.image("https://www.w3schools.com/howto/img_avatar.png")

    st_c_chat = st.container(border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                if user_image:
                    st_c_chat.chat_message(msg["role"],avatar=user_image).markdown((msg["content"]))
                else:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            elif msg["role"] == "assistant":
                st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            else:
                try:
                    image_tmp = msg.get("image")
                    if image_tmp:
                        st_c_chat.chat_message(msg["role"],avatar=image_tmp).markdown((msg["content"]))
                except:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))

    def generate_response(prompt):
        prompt = prompt.lower().strip()

        # Greeting
        if re.search(r"\b(hi|hello|hey|what can you do|who are you|help)\b", prompt):
            return ("Hello! Here we have a couple of pages you might want to check out:\n"
                    "- 2D Visualization of Word Embeddings\n"
                    "- 3D Visualization of Word Embeddings\n"
                    "- Skip-Gram Word2Vec\n"
                    "- Comparison of SKIP-GRAM and CBOW Word2Vec Models\n"
                    "You can navigate the pages yourself through the sidebar on the left-hand side. Or you can ask me for help."
                    )

        # 2D Visualization page
        if "2d" in prompt:
            return ("Kindly check out the page titled \"2D\". You can navigate the pages through the sidebar on the left-hand side.")

        # 3D Visualization page
        if "3d" in prompt:
            return ("Kindly check out the page titled \"3D\". You can navigate the pages through the sidebar on the left-hand side.")

        # Skip-gram Word2Vec: With and Without Stopwords
        if "skip gram" in prompt or "skip-gram" in prompt:
            return ("Kindly check out the page titled \"SKIP-GRAM\". You can navigate the pages through the sidebar on the left-hand side.")

        # Comparison of Skip-gram and CBOW Word2Vec Models
        if "cbow" in prompt or "comparison" in prompt:
            return ("Kindly check out the page titled \"CBOW\". You can navigate the pages through the sidebar on the left-hand side.")

        # If not recognized
        return ("I'm not sure what you mean. as I'm still stupid.\n")

    # Chat function section (timing included inside function)
    def chat(prompt: str):
        st_c_chat.chat_message("user",avatar=user_image).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = generate_response(prompt)
        # response = f"You type: {prompt}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st_c_chat.chat_message("assistant").write_stream(stream_data(response))

    
    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)

if __name__ == "__main__":
    main()