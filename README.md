Dense Retrieval：Sentence‑BERT 產生商品向量 -> 初步候選集
Two‑Tower Model：使用者向量 vs 商品向量，對比式學習提升匹配能力
BERT4Rec：根據使用者購物序列，重新排序候選商品

Stage1 Dense Retrieval（Sentence-BERT）
程式碼片段：
dense_model = SentenceTransformer("msmarco-distilbert-base-v4") item_embeddings = dense_model.encode(products["text"]) 
輸入：商品文字
輸出：item embedding (768-d vector)
輸入：每個商品的文字資訊（商品名稱 + Aisle + Department）
輸出：商品向量 (embedding)
使用 cosine similarity 找出前 200 個相似商品，作為候選集
Stage 2: Two-Tower Dual Encoder
scores = torch.matmul(emb_u, emb_i.T) loss = F.cross_entropy(scores, labels) 
Training pair：
user tower : previous items item tower : last purchased item 
Loss：MultipleNegativesRankingLoss
✅ 讓 user embedding “靠近” user 最後真的買的那個商品
Item Tower：編碼商品資訊
採用對比式學習（Contrastive Learning）拉近正樣本距離
Loss：MultipleNegativesRankingLoss

Stage 3: BERT4Rec (Ranking)
loss = bert_model(input_ids, attention_mask, labels)
輸入：使用者購物序列 "item1 [SEP] item2 [SEP] item3"
輸出：預測下一個 item
使用 Transformer 去預測「下一個商品」
輸入：使用者歷史購物序列
輸出：對所有商品的 logits 分數(回歸)，重新排序候選集


Three-Stage Recommendation Pipeline
Dense Retrieval → Two-Tower Encoder → BERT4Rec Ranking

