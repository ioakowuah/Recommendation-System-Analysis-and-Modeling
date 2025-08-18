import streamlit as st
import pickle
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download, login



# --- Hugging Face Repo Info ---
REPO_ID = "ioakowuah/Recommendation-System-Analysis-and-Modeling"   # without https://
REPO_TYPE = "model"  # or "model" depending where you uploaded

MODEL_FILES = {
    "svd_model": "svd_model.pkl",
    "user_factors": "user_factors.pkl",
    "item_factors": "item_factors.pkl",
    "user2idx": "user2idx.pkl",
    "idx2item": "idx2item.pkl",
    "unique_users": "unique_users.csv"
}

@st.cache_resource
def load_models():
    svd_model = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["svd_model"], repo_type=REPO_TYPE), "rb"))
    user_factors = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["user_factors"], repo_type=REPO_TYPE), "rb"))
    item_factors = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["item_factors"], repo_type=REPO_TYPE), "rb"))
    user2idx = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["user2idx"], repo_type=REPO_TYPE), "rb"))
    idx2item = pickle.load(open(hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["idx2item"], repo_type=REPO_TYPE), "rb"))

    unique_users = pd.read_csv(
        hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["unique_users"], repo_type=REPO_TYPE)
    )["user_id"].tolist()

    return svd_model, user_factors, item_factors, user2idx, idx2item, unique_users




