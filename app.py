import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page settings
st.set_page_config(page_title="AI Job Matcher", layout="wide")

st.title("🎯 AI Job Recommendation System")
st.write("Enter your skills to find the most suitable job roles.")

# Load dataset
try:
    df = pd.read_csv('processed_jobs.csv')
except:
    st.error("processed_jobs.csv not found. Run preprocess.py first.")
    st.stop()

# User input
user_skills = st.text_input(
    "Enter your skills",
    placeholder="Example: python, machine learning, data analysis"
)

# Main recommendation system
if user_skills:

    user_skills = user_skills.lower()

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english')

    all_texts = pd.concat(
        [pd.Series([user_skills]), df['tags']],
        ignore_index=True
    )

    tfidf_matrix = tfidf.fit_transform(all_texts)

    user_vector = tfidf_matrix[0:1]
    job_vectors = tfidf_matrix[1:]

    # Cosine similarity
    similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()

    df['Match Score'] = similarity_scores

    # Get top results
    top_jobs = df.sort_values(by='Match Score', ascending=False).head(5)

    if top_jobs['Match Score'].max() < 0.05:
        st.warning("No strong matches found. Try adding more skills.")
        st.stop()

    st.subheader("💼 Top Job Recommendations")

    # Display jobs
    for index, row in top_jobs.iterrows():

        with st.container():

            col1, col2 = st.columns([3,1])

            with col1:
                st.markdown(f"### {row['Job Title'].title()}")
                st.write(f"Industry: {row['Industry'].title()}")
                st.write(f"Role Category: {row['Role Category'].title()}")
                st.write(f"Key Skills: {row['Key Skills']}")

            with col2:
                score = round(row['Match Score']*100,1)
                st.metric("Match Score", f"{score}%")

            # Skill gap analysis
            user_set = set(user_skills.split(", "))
            job_set = set(row['Key Skills'].split(", "))

            missing = job_set - user_set

            if missing:
                st.info("Recommended skills to learn: " + ", ".join(list(missing)[:5]))

            st.divider()

    