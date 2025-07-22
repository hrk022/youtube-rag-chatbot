import os
import streamlit as st
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import uuid

# ---- Load API Keys ----
# It's good practice to handle the case where the .env file might not be found.
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
# Set environment variables for LangChain

os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# ---- Streaming Token Callback Handler ----
class StreamHandler(BaseCallbackHandler):
    """Callback handler to stream LLM responses to a Streamlit container."""
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Append new tokens to the text and update the container."""
        self.text += token
        self.container.markdown(self.text + "▌")
    
    def on_llm_end(self, *args, **kwargs) -> None:
        """Final update to the container with the complete text."""
        self.container.markdown(self.text)

# ---- Streamlit App Config ----
st.set_page_config(page_title="🎥 Chat with YouTube Video", layout="centered")
st.title("🎥 Chat with a YouTube Video Transcript")

# ---- Session State Initialization ----
# Initialize session state variables if they don't exist.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ---- Helper Functions ----
def extract_video_id(url: str):
    """Extracts the YouTube video ID from various URL formats."""
    if not url:
        return None
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            return parse_qs(query.query).get('v', [None])[0]
        if query.path.startswith(('/embed/', '/v/')):
            return query.path.split('/')[2]
    return None

def process_transcript(video_id: str):
    """Fetches transcript, creates chunks, and builds a FAISS vector store."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join(entry["text"] for entry in transcript_list)
        
        document = Document(page_content=full_text)
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents([document])

        if not chunks:
            st.error("❌ Could not create text chunks from the transcript.")
            return None

        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
        
        return vectorstore.as_retriever()

    except Exception as e:
        st.error(f"❌ Failed to fetch or process transcript: {e}")
        return None

# ---- Custom Prompt Template ----
# This new prompt template instructs the model to provide a detailed answer.
prompt_template = """Use the following pieces of context to answer the question at the end. Provide a detailed and comprehensive response based on the context. Explain the key points and elaborate on the information available.

Context: {context}

Question: {question}
Helpful and Detailed Answer:"""

QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# ---- Main App Logic ----
with st.sidebar:
    st.header("Video Source")
    youtube_url = st.text_input("🎬 Enter YouTube URL:")

    if st.button("Process Video"):
        if youtube_url:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                st.error("❌ Invalid YouTube URL. Please try again.")
            else:
                with st.spinner("⏳ Processing YouTube transcript..."):
                    retriever = process_transcript(video_id)
                    if retriever:
                        llm = ChatOpenAI(
                            model_name="llama3-70b-8192",
                            temperature=0.7,
                            openai_api_key=os.getenv("OPENAI_API_KEY"),
                            openai_api_base="https://api.groq.com/openai/v1",
                            streaming=True
                        )
                        memory = ConversationBufferMemory(
                            memory_key="chat_history", 
                            return_messages=True
                        )
                        # Pass the custom prompt to the chain
                        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=retriever,
                            memory=memory,
                            return_source_documents=False,
                            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
                        )
                        st.success("✅ Video processed! You can now ask questions.")
                        st.session_state.chat_history = []
        else:
            st.warning("Please enter a YouTube URL.")

# ---- Chat Interface ----
for sender, message in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(message)

if st.session_state.qa_chain:
    if user_query := st.chat_input("Ask something about the video..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append(("user", user_query))

        with st.chat_message("assistant"):
            response_container = st.empty()
            stream_handler = StreamHandler(response_container)
            
            result = st.session_state.qa_chain.invoke(
                {"question": user_query},
                config={"callbacks": [stream_handler]}
            )
            
            final_answer = result.get("answer", "Sorry, I encountered an error.")
            st.session_state.chat_history.append(("assistant", final_answer))
else:
    st.info("Please enter a YouTube URL in the sidebar and click 'Process Video' to begin.")
