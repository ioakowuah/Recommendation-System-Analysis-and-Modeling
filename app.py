import streamlit as st
import pickle
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# --- Hugging Face Repo Info ---
REPO_ID = "https://huggingface.co/datasets/ioakowuah/RecommendationSystem"   # e.g. "nana-baffour/recommender-model"
MODEL_FILES = {
    "svd_model": "svd_model.pkl",
    "user_factors": "user_factors.pkl",
    "item_factors": "item_factors.pkl",
    "user2idx": "user2idx.pkl",
    "idx2item": "idx2item.pkl",
    "unique_users": "unique_users.csv"
}

# --- Load Files from Hugging Face ---
@st.cache_resource
def load_models():
    svd_model = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["svd_model"]), "rb"))
    user_factors = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["user_factors"]), "rb"))
    item_factors = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["item_factors"]), "rb"))
    user2idx = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["user2idx"]), "rb"))
    idx2item = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["idx2item"]), "rb"))

    # Unique users as DataFrame
    unique_users = pd.read_csv(
        hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["unique_users"])
    )["user_id"].tolist()

    return svd_model, user_factors, item_factors, user2idx, idx2item, unique_users

svd_model, user_factors, item_factors, user2idx, idx2item, unique_users = load_models()

# --- Sample only 20 users for UI ---
if len(unique_users) > 20:
    sampled_users = np.random.choice(unique_users, 20, replace=False)
else:
    sampled_users = unique_users

# --- Recommendation Function ---
def recommend_for_user(user_id, top_n=10):
    if user_id not in user2idx:
        return []

    user_index = user2idx[user_id]
    scores = np.dot(item_factors, user_factors[user_index])
    top_indices = np.argsort(scores)[::-1][:top_n]
    recommended_items = [idx2item[i] for i in top_indices]
    return recommended_items

# --- Streamlit UI ---
st.title("🔎 Recommendation System Demo (Hybrid via Hugging Face)")

st.write("Select a user from the dropdown to see personalized recommendations.")

selected_user = st.selectbox("Choose a User ID", sampled_users)

if st.button("Get Recommendations"):
    recs = recommend_for_user(selected_user, top_n=10)
    if recs:
        st.success(f"Top recommendations for User {selected_user}:")
        for i, item in enumerate(recs, start=1):
            st.write(f"{i}. Item ID: {item}")
    else:
        st.warning("No recommendations available for this user.")

