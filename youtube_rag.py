import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import uuid
import os

# ---- Streaming Token Callback Handler ----
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")  # typing cursor effect

# ---- Streamlit App Config ----
st.set_page_config(page_title="🎥 Chat with YouTube Video")
st.title("🎥 Chat with a YouTube Video Transcript")

# ---- YouTube Video ID Extractor ----
def extract_video_id(url):
    query = urlparse(url)
    if query.hostname == "youtu.be":
        return query.path[1:]
    if query.hostname in ['www.youtube.com', 'youtube.com']:
        if query.path == "/watch":
            return parse_qs(query.query).get('v', [None])[0]
        if query.path.startswith("/embed/") or query.path.startswith("/v/"):
            return query.path.split("/")[2]
    return None

# ---- Session State Initialization ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ---- Transcript Processing & Vector Store ----
def process_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        st.error(f"Failed to fetch the transcript: {e}")
        return None

    full_text = " ".join(entry["text"] for entry in transcript)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([full_text])

    embeddings = HuggingFaceEmbeddings()
    vector_dir = f"youtube_vector/{uuid.uuid4()}"
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=vector_dir)
    vectorstore.persist()
    return vectorstore.as_retriever()

# ---- YouTube URL Input ----
youtube_url = st.text_input("🎬 Enter the URL of the YouTube video:")

if youtube_url and st.session_state.qa_chain is None:
    video_id = extract_video_id(youtube_url)
    if video_id:
        with st.spinner("⏳ Processing YouTube transcript..."):
            retriever = process_transcript(video_id)
            if retriever:
                llm = ChatOllama(model="llama3:latest", streaming=True, temperature=0.7)
                st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                st.success("✅ Ready! Ask anything about the video 👇")
            else:
                st.error("❌ Could not process the video. Please check the URL or try another.")
    else:
        st.error("⚠️ Invalid YouTube URL.")

# ---- Chat Interface ----
if st.session_state.qa_chain:
    user_query = st.chat_input("Ask something about the video...")
    if user_query:
        st.session_state.chat_history.append(("user", user_query))
        st.chat_message("user").markdown(user_query)

        with st.chat_message("assistant"):
            stream_container = st.empty()
            handler = StreamHandler(stream_container)
            result = st.session_state.qa_chain.run(user_query, callbacks=[handler])
            st.session_state.chat_history.append(("assistant", result))

# ---- Show Chat History ----
for sender, msg in st.session_state.chat_history[:-1]:
    st.chat_message(sender).markdown(msg)
