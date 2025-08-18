import streamlit as st
import pickle
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# --- Hugging Face Repo Info ---
REPO_ID = "ioakowuah/Recommendation-System-Analysis-and-Modeling"
REPO_TYPE = "model"

MODEL_FILES = {
    "svd_model": "svd_model.pkl",
    "user_factors": "user_factors.pkl",
    "item_factors": "item_factors.pkl",
    "user2idx": "user2idx.pkl",
    "idx2item": "idx2item.pkl",
}

@st.cache_resource
def load_models():
    try:
        svd_model = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["svd_model"], repo_type=REPO_TYPE), "rb"))
        user_factors = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["user_factors"], repo_type=REPO_TYPE), "rb"))
        item_factors = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["item_factors"], repo_type=REPO_TYPE), "rb"))
        user2idx = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["user2idx"], repo_type=REPO_TYPE), "rb"))
        idx2item = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["idx2item"], repo_type=REPO_TYPE), "rb"))

        # Build unique_users directly from user2idx keys
        unique_users = list(user2idx.keys())

        return svd_model, user_factors, item_factors, user2idx, idx2item, unique_users, False

    except Exception as e:
        st.error(f"❌ Failed to load models from Hugging Face: {e}")
        return None, None, None, None, None, None, True  # flag = True for fallback

# --- Main UI ---
st.title("🔎 Recommendation System Demo")
st.write("This app loads a trained SVD recommender from Hugging Face Hub. If loading fails, a demo dataset is used.")

svd_model, user_factors, item_factors, user2idx, idx2item, unique_users, fallback = load_models()

if fallback:
    # --- DEMO fallback data ---
    st.warning("⚠️ Using demo dataset because Hugging Face model could not be loaded.")
    unique_users = [f"user_{i}" for i in range(1, 6)]   # 5 fake users
    items = [f"item_{i}" for i in range(1, 21)]         # 20 fake items
    user2idx = {u: i for i, u in enumerate(unique_users)}
    idx2item = {i: it for i, it in enumerate(items)}
    user_factors = np.random.rand(len(unique_users), 10)
    item_factors = np.random.rand(len(items), 10)

# --- Recommendation UI ---
if unique_users is not None and len(unique_users) > 0:
    sampled_users = np.random.choice(unique_users, min(20, len(unique_users)), replace=False)
    selected_user = st.selectbox("Choose a User ID", sampled_users)

    if st.button("Get Recommendations"):
        if selected_user not in user2idx:
            st.warning("User not found in model.")
        else:
            user_index = user2idx[selected_user]
            scores = np.dot(item_factors, user_factors[user_index])
            top_indices = np.argsort(scores)[::-1][:10]
            recs = [idx2item[i] for i in top_indices]

            st.success(f"Top recommendations for User {selected_user}:")
            for i, item in enumerate(recs, start=1):
                st.write(f"{i}. Item ID: {item}")
else:
    st.error("🚨 No users available.")
