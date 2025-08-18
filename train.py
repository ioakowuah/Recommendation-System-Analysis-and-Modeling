import pandas as pd
import numpy as np
import pickle
import json
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD

# --- Load your dataset ---
# Replace this with your actual events dataset
events_df = pd.read_csv("events.csv")

# Map event types to rating scores
event_score = {"view": 1, "addtocart": 3, "transaction": 5}
events_df["rating"] = events_df["event"].map(event_score)

# Get unique users and items
unique_users = events_df["visitorid"].unique()
unique_items = events_df["itemid"].unique()
print("Unique users:", len(unique_users))
print("Unique items:", len(unique_items))

# Create ID→index mappings
user2idx = {user: idx for idx, user in enumerate(unique_users)}
item2idx = {item: idx for idx, item in enumerate(unique_items)}
idx2item = {idx: item for item, idx in item2idx.items()}

# Map IDs to integer indices
events_df["user_idx"] = events_df["visitorid"].map(user2idx)
events_df["item_idx"] = events_df["itemid"].map(item2idx)

# Build sparse user–item matrix
rows = events_df["user_idx"].to_numpy()
cols = events_df["item_idx"].to_numpy()
data = events_df["rating"].to_numpy()

user_item_matrix = coo_matrix(
    (data, (rows, cols)),
    shape=(len(unique_users), len(unique_items))
)

# --- Train SVD ---
n_components = 50
svd_model = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd_model.fit_transform(user_item_matrix)  # (num_users, n_components)
item_factors = svd_model.components_.T                   # (num_items, n_components)

print("SVD model trained.")
print("User factors shape:", user_factors.shape)
print("Item factors shape:", item_factors.shape)

# --- Save Pickle Files ---
with open("svd_model.pkl", "wb") as f:
    pickle.dump(svd_model, f)
with open("user_factors.pkl", "wb") as f:
    pickle.dump(user_factors, f)
with open("item_factors.pkl", "wb") as f:
    pickle.dump(item_factors, f)
with open("user2idx.pkl", "wb") as f:
    pickle.dump(user2idx, f)
with open("idx2item.pkl", "wb") as f:
    pickle.dump(idx2item, f)

# --- Save Mappings for Easier Loading ---
# Users
pd.DataFrame({"user_id": unique_users}).to_csv("unique_users.csv", index=False)

# Items
pd.DataFrame({"item_id": unique_items}).to_csv("unique_items.csv", index=False)

# JSON mappings (optional, easier for APIs)
with open("mappings.json", "w") as f:
    json.dump({"user2idx": user2idx, "idx2item": idx2item}, f)

print("✅ Model and mappings saved successfully!")
