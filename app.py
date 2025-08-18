import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Load Saved Models & Mappings ---
@st.cache_resource
def load_models():
    with open("svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)
    with open("user_factors.pkl", "rb") as f:
        user_factors = pickle.load(f)
    with open("item_factors.pkl", "rb") as f:
        item_factors = pickle.load(f)
    with open("user2idx.pkl", "rb") as f:
        user2idx = pickle.load(f)
    with open("idx2item.pkl", "rb") as f:
        idx2item = pickle.load(f)
    return svd_model, user_factors, item_factors, user2idx, idx2item

svd_model, user_factors, item_factors, user2idx, idx2item = load_models()

# Load unique users file
unique_users = pd.read_csv("unique_users.csv")["user_id"].tolist()

# Sample only 20 random user IDs for better performance
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
st.title("🔎 Recommendation System Demo")

st.write("Select a user from the dropdown to see personalized recommendations.")

# Dropdown for user selection (limited to 20)
selected_user = st.selectbox("Choose a User ID", sampled_users)

if st.button("Get Recommendations"):
    recs = recommend_for_user(selected_user, top_n=10)
    if recs:
        st.success(f"Top recommendations for User {selected_user}:")
        for i, item in enumerate(recs, start=1):
            st.write(f"{i}. Item ID: {item}")
    else:
        st.warning("No recommendations available for this user.")
