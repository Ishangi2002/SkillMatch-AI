import os
import streamlit as st
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Add it to your .env file.")
    st.stop()

# Initialize the modern Google GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)

# Use the 2026 stable workhorse model
MODEL_ID = "gemini-2.5-flash" 

# -------------------------------
# Streamlit page settings
# -------------------------------
st.set_page_config(page_title="RAG Job Matcher", layout="wide")
st.title("🤖 RAG-Based AI Job Recommendation System")

# -------------------------------
# Load vector store and resources
# -------------------------------
@st.cache_resource
def load_resources():
    try:
        # Load FAISS index
        if not os.path.exists("faiss_index.bin"):
            st.error("FAISS index file not found!")
            st.stop()
        index = faiss.read_index("faiss_index.bin")
        
        # Load Job Metadata
        with open("job_metadata.pkl", "rb") as f:
            df = pickle.load(f)
            
        # Load Embedding Model
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        return index, df, embed_model
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

index, df, embed_model = load_resources()

# -------------------------------
# User input
# -------------------------------
user_skills = st.text_input(
    "Enter your skills",
    placeholder="Example: python, machine learning, data analysis"
)

# -------------------------------
# RAG pipeline
# -------------------------------
if user_skills:

    # 1️⃣ Convert user query to embedding
    query_embedding = embed_model.encode([user_skills])

    # 2️⃣ Retrieve top jobs
    k = 5
    distances, indices = index.search(query_embedding, k)
    top_jobs = df.iloc[indices[0]]

    st.subheader("💼 Top Job Matches")

    # Display retrieved jobs
    for _, row in top_jobs.iterrows():
        with st.container():
            st.markdown(f"### {row['Job Title'].title()}")
            st.write(f"**Industry:** {row['Industry'].title()}")
            st.write(f"**Role Category:** {row['Role Category'].title()}")
            st.write(f"**Key Skills:** {row['Key Skills']}")
            st.divider()

    # 3️⃣ Prepare prompt for LLM
    job_info_text = ""
    for _, row in top_jobs.iterrows():
        job_info_text += (
            f"Job Title: {row['Job Title']}, "
            f"Skills: {row['Key Skills']}, "
            f"Industry: {row['Industry']}\n"
        )

    prompt = f"""
    User skills:
    {user_skills}

    Recommended jobs:
    {job_info_text}

    Explain why these jobs match the user.
    Suggest missing skills the user should learn.
    Give brief career advice.
    """

    # 4️⃣ Generate AI explanation using the updated model ID
    with st.spinner("🧠 AI is analyzing your career path..."):
        try:
            # The client.models.generate_content is the standard for 2026
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt
            )
            
            if response.text:
                st.subheader("🧠 AI Career Advisor")
                st.write(response.text)
            else:
                st.warning("AI produced an empty response. Check your API safety settings.")
                
        except Exception as e:
            # Handle the 404 by providing a specific hint if it happens again
            if "404" in str(e):
                st.error("Model ID not found. Your API key might not have access to Gemini 2.5 Flash yet.")
                st.info("Try changing MODEL_ID to 'gemini-3-flash-preview' in your code.")
            else:
                st.error(f"AI Generation Error: {e}")