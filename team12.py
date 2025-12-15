# =============================================
# Instacart Dense Retrieval (Dual Encoder + Sentence-BERT)
# Evaluate Cold/Warm/Active users by Recall@K and NDCG@K
# =============================================

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# 1️⃣ Dataset paths (use your style)
# ===============================
DATA_DIR = "./instacart_data"  # <- 確認資料夾名稱正確
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
AISLES_CSV = os.path.join(DATA_DIR, "aisles.csv")
DEPTS_CSV = os.path.join(DATA_DIR, "departments.csv")
ORDERS_CSV = os.path.join(DATA_DIR, "orders.csv")
OP_PRIOR_CSV = os.path.join(DATA_DIR, "order_products__prior.csv")
OP_TRAIN_CSV = os.path.join(DATA_DIR, "order_products__train.csv")

# ===============================
# 2️⃣ Load data
# ===============================
print("Loading data ...")
orders = pd.read_csv(ORDERS_CSV)
order_products = pd.read_csv(OP_PRIOR_CSV)
products = pd.read_csv(PRODUCTS_CSV)
aisles = pd.read_csv(AISLES_CSV)
departments = pd.read_csv(DEPTS_CSV)

# enrich product info
products = (
    products
    .merge(aisles, on="aisle_id", how="left")
    .merge(departments, on="department_id", how="left")
)
products["text"] = (
    products["product_name"].fillna("") + " " +
    products["aisle"].fillna("") + " " +
    products["department"].fillna("")
)

# join user-item history
user_orders = (
    orders.merge(order_products, on="order_id")
    .merge(products[["product_id", "text"]], on="product_id")
)

# ===============================
# 3️⃣ Split user groups
# ===============================
user_counts = user_orders.groupby("user_id")["product_id"].nunique()
cold_users = user_counts[user_counts < 5].index
warm_users = user_counts[(user_counts >= 5) & (user_counts < 20)].index
active_users = user_counts[user_counts >= 50].index

groups = {
    "cold": cold_users,
    "warm": warm_users,
    "active": active_users
}

# ===============================
# 4️⃣ Encode items (Sentence-BERT)
# ===============================
print("Encoding item embeddings (Sentence-BERT)...")
model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v4")
item_embeddings = model.encode(
    products["text"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)
product_id_to_idx = dict(zip(products["product_id"], range(len(products))))

# ===============================
# 5️⃣ Helper functions
# ===============================
def get_user_embedding(user_id):
    """User embedding = average of their purchased item embeddings"""
    user_items = user_orders[user_orders["user_id"] == user_id]["product_id"].values
    idx = [product_id_to_idx[i] for i in user_items if i in product_id_to_idx]
    if len(idx) == 0:
        return np.zeros(item_embeddings.shape[1])
    return np.mean(item_embeddings[idx], axis=0)

def recall_at_k(y_true, y_pred, k):
    return len(set(y_true) & set(y_pred[:k])) / len(set(y_true)) if len(y_true) else 0

def ndcg_at_k(y_true, y_pred, k):
    dcg = 0.0
    for i, p in enumerate(y_pred[:k]):
        if p in y_true:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(y_true), k)))
    return dcg / idcg if idcg > 0 else 0

# ===============================
# 6️⃣ Evaluate per group
# ===============================
K_values = [5, 10, 20]
results = {}

for group_name, user_ids in groups.items():
    print(f"\nEvaluating group: {group_name} ({len(user_ids)} users)")
    recalls = {k: [] for k in K_values}
    ndcgs = {k: [] for k in K_values}

    for uid in tqdm(user_ids):
        user_embed = get_user_embedding(uid).reshape(1, -1)
        sims = cosine_similarity(user_embed, item_embeddings)[0]
        top_items = np.argsort(-sims)[:50]
        top_product_ids = products.iloc[top_items]["product_id"].tolist()
        ground_truth = user_orders[user_orders["user_id"] == uid]["product_id"].unique()

        for k in K_values:
            recalls[k].append(recall_at_k(ground_truth, top_product_ids, k))
            ndcgs[k].append(ndcg_at_k(ground_truth, top_product_ids, k))

    results[group_name] = {
        "Recall": {k: np.mean(recalls[k]) for k in K_values},
        "NDCG": {k: np.mean(ndcgs[k]) for k in K_values},
    }

# ===============================
# 7️⃣ Output results as table
# ===============================
print("\n=== Final Evaluation Results ===")
for group, metrics in results.items():
    print(f"\n{group.upper()}")
    print(pd.DataFrame(metrics).round(5))
