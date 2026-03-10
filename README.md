# 📄 RAG-Based AI Job Recommendation System – Skill-Based Job Recommender

**RAG-Based AI Job Recommendation System** is an AI-powered tool that analyzes a user’s skills and provides personalized job recommendations. It uses **Retrieval-Augmented Generation (RAG)** with embeddings to intelligently match user skills with relevant jobs and provides **career guidance** using a Large Language Model (Gemini API). The interactive web interface is built with **Streamlit**.

---

## Features

* ✍️ Enter skills via a simple web interface
* 🔎 Retrieves top job recommendations based on embedding similarity
* 🧠 Provides AI-generated explanations, missing skills suggestions, and career advice
* 🛠️ Lightweight and interactive Streamlit interface
* 📊 Easily extendable with a custom job dataset

---

## How the Project Works

1. User inputs a list of skills through the Streamlit interface.
2. Skills are converted into **vector embeddings** using `SentenceTransformer`.
3. **FAISS** searches the job vector database to retrieve the top matching jobs.
4. Job details are prepared and passed to a **Large Language Model (Gemini API)** for analysis.
5. The system generates explanations, missing skill recommendations, and career advice.
6. The top jobs and AI guidance are displayed interactively in the app.

---

## Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Embeddings & Matching:** `SentenceTransformer`, FAISS
* **AI Career Advisor:** Gemini API (`google.generativeai`)
* **Data Handling:** pandas
* **Dataset:** CSV file containing Key Skills, Job Title, Role Category, Functional Area, Industry

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Ishangi2002/SkillMatch-AI.git
cd rag-job-recommender
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Create a .env file in the project root:

```bash
GEMINI_API_KEY=your_real_gemini_api_key_here
```

4. Ensure the FAISS index and job metadata are built:

```bash
python preprocess.py
python build_vector_store.py
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```
2. Enter your skills (comma-separated) in the input box.

3. Press Enter.

4. View the top recommended jobs along with AI-generated career advice.