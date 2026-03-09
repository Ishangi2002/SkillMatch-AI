# 📄 AI Job Recommendation System – Skill-Based Job Recommender

**AI Job Recommendation System** is an AI-powered tool that analyzes a user’s skills and provides personalized job recommendations. The system uses Natural Language Processing (NLP) techniques like TF-IDF and cosine similarity to intelligently match user skills with relevant jobs in the dataset, delivering career guidance through an interactive web interface built with Streamlit.

---

## Features

* ✍️Enter skills via a simple web interface
* 📑Matches user skills with job descriptions using TF-IDF & cosine similarity
* 🔎Displays top job recommendations based on skill match
* 🛠️Lightweight and interactive Streamlit interface
* 📊Easily extendable with a custom CSV job dataset

---

## How the Project Works

1. User inputs a list of skills through the Streamlit interface.
2. The system vectorizes the user input and job descriptions using TF-IDF.
3. Cosine similarity is calculated between the user skills and all jobs in the dataset.
4. Jobs are ranked based on similarity scores.
5. The top matching jobs are displayed to the user with their titles and descriptions.

---

## Tech Stack

* Frontend: Streamlit
* Backend: Python
* NLP & Matching: TF-IDF, Cosine Similarity (scikit-learn)
* Data Handling: pandas
* Dataset: CSV file containing Key Skills,Job Title,Role Category,Functional Area,Industry

---

## Usage

1. Open the Streamlit app in your browser:

   ```bash
   streamlit run app.py
   ```
2. Enter your skills (comma-separated) in the input box.
3. Click Enter.
4. View the top recommended jobs based on your skill set.




