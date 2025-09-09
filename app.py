# app_with_images.py
import streamlit as st
import pandas as pd
import joblib
from rapidfuzz import process
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Load cleaned dataset & feature matrix
# ---------------------------
df = pd.read_csv("books_dataset_cleaned_with_img.csv")
final_matrix = joblib.load("final_matrix.pkl")

# Load or compute similarity matrix
try:
    similarity_matrix = joblib.load("similarity_matrix.pkl")
except:
    similarity_matrix = cosine_similarity(final_matrix, dense_output=True)
    joblib.dump(similarity_matrix, "similarity_matrix.pkl")

# ---------------------------
# Recommender function
# ---------------------------


def recommend_books(title, n=5):
    """
    Recommend top-N books similar to the given title with images.
    Uses fuzzy matching for flexible search.
    """
    # Fuzzy match
    best_match = process.extractOne(title, df['title'], score_cutoff=60)
    if not best_match:
        return pd.DataFrame(), f"No book found matching '{title}'"

    matched_title = best_match[0]
    idx = df[df['title'] == matched_title].index[0]

    # Similarity scores
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[
        1:n+1]  # exclude self
    book_indices = [i[0] for i in sim_scores]

    # Recommended books
    recommendations = df.iloc[book_indices][[
        'title', 'author', 'genre', 'rating', 'img']].reset_index(drop=True)
    return recommendations, matched_title


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📚 Book Recommender System with Covers")
st.write("Type a book title and get top similar book recommendations with cover images!")

# User input
book_input = st.text_input("Enter a book title:")

if book_input:
    recommendations, matched_title = recommend_books(book_input, n=5)
    if recommendations.empty:
        st.warning(f"No recommendations found for '{book_input}'")
    else:
        st.success(f"Recommendations for: {matched_title}")
        for idx, row in recommendations.iterrows():
            st.markdown(f"**{row['title']}** by {row['author']}")
            st.write(f"Genre: {row['genre']}")
            st.write(f"Rating: {row['rating']}")
            st.image(row['img'], width=120)
            st.markdown("---")
