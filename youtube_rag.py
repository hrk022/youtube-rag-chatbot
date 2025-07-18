import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs
import uuid
import random

# ---- Proxies for YouTube Transcript API ----
PROXIES = [
    "http://103.78.96.246:8080",
    "http://194.87.102.146:3128",
    # Add more free HTTP proxies here
]

# ---- Authenticated Proxies ----
AUTH_PROXIES = [
    {
        "http": "http://lqmgbesh:b61o8yc2ek4f@38.154.227.167:5868",
        "https": "http://lqmgbesh:b61o8yc2ek4f@38.154.227.167:5868"
    },
    {
        "http": "http://lqmgbesh:b61o8yc2ek4f@23.95.150.145:6114",
        "https": "http://lqmgbesh:b61o8yc2ek4f@23.95.150.145:6114"
    }
]

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

# ---- Fetch Transcript with Proxy Rotation ----
# ---- Fetch Transcript using Free & Authenticated Proxies ----
def fetch_transcript(video_id):
    proxy_pool = [{"http": p, "https": p} for p in PROXIES] + AUTH_PROXIES

    for proxy in proxy_pool:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, proxies=proxy)
            return transcript
        except (TranscriptsDisabled, NoTranscriptFound):
            st.error("Transcript not available for this video.")
            return None
        except Exception:
            st.warning(f"Proxy failed: {proxy['http']}")

    # Try direct connection if all proxies fail
    try:
        st.info("Trying direct connection...")
        return YouTubeTranscriptApi.get_transcript(video_id)
    except Exception:
        st.error("All proxies failed. Direct connection failed too.")
        return None

# ---- Transcript Processing & Vector Store ----

def process_transcript(video_id):
    transcript = fetch_transcript(video_id)  # <- now using full proxy logic
    if not transcript:
        return None

    full_text = " ".join(entry["text"] for entry in transcript)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([full_text])

    embeddings = HuggingFaceEmbeddings()
    vector_dir = f"youtube_vector/{uuid.uuid4()}"
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=vector_dir)
    vectorstore.persist()
    return vectorstore.as_retriever()

# ---- Session State Initialization ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

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
                st.error("❌ Could not process the video. Please try another.")
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
