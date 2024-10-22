from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
model = BertModel.from_pretrained('google-bert/bert-base-uncased')

# 定义GT, chosen, rejected句子
gt = "The players are maneuvering around the court to avoid being hit by the balls."
chosen = "The players are actively moving around the court, throwing and dodging balls."
rejected = "The players appear to be taking evasive action to avoid being hit by the balls. They are positioning themselves strategically and moving to maintain a safe distance from the balls as they are thrown or hit."

# 对GT, chosen, rejected句子进行BERT编码
inputs_gt = tokenizer(gt, return_tensors='pt', max_length=128, truncation=True, padding=True)
inputs_chosen = tokenizer(chosen, return_tensors='pt', max_length=128, truncation=True, padding=True)
inputs_rejected = tokenizer(rejected, return_tensors='pt', max_length=128, truncation=True, padding=True)

# 通过BERT模型获取句子向量
with torch.no_grad():
    outputs_gt = model(**inputs_gt)
    outputs_chosen = model(**inputs_chosen)
    outputs_rejected = model(**inputs_rejected)

# 获取每个句子的[CLS]向量，作为句子表示
gt_embedding = outputs_gt.last_hidden_state[:, 0, :].numpy()
chosen_embedding = outputs_chosen.last_hidden_state[:, 0, :].numpy()
rejected_embedding = outputs_rejected.last_hidden_state[:, 0, :].numpy()

# 计算GT和chosen, GT和rejected的BERT相似度
bert_similarity_gt_chosen = cosine_similarity(gt_embedding, chosen_embedding)[0][0]
bert_similarity_gt_rejected = cosine_similarity(gt_embedding, rejected_embedding)[0][0]

print(f"BERT similarity (GT vs Chosen): {bert_similarity_gt_chosen:.4f}")
print(f"BERT similarity (GT vs Rejected): {bert_similarity_gt_rejected:.4f}")

# 使用词袋模型计算相似度
vectorizer = CountVectorizer()

X = vectorizer.fit_transform([gt, chosen, rejected])

# 计算GT和chosen, GT和rejected的WordBag相似度
similarity_matrix = cosine_similarity(X)

wordbag_similarity_gt_chosen = similarity_matrix[0][1]
wordbag_similarity_gt_rejected = similarity_matrix[0][2]

print(f"WordBag similarity (GT vs Chosen): {wordbag_similarity_gt_chosen:.4f}")
print(f"WordBag similarity (GT vs Rejected): {wordbag_similarity_gt_rejected:.4f}")

# 结合BERT和WordBag特征，进行特征拼接后的相似度计算
bow_vector_gt = X.toarray()[0]
bert_vector_gt = gt_embedding.flatten()

bow_vector_chosen = X.toarray()[1]
bert_vector_chosen = chosen_embedding.flatten()

bow_vector_rejected = X.toarray()[2]
bert_vector_rejected = rejected_embedding.flatten()

# GT和chosen的组合向量
combined_vector_gt_chosen_1 = np.concatenate([bow_vector_gt, bert_vector_gt])
combined_vector_gt_chosen_2 = np.concatenate([bow_vector_chosen, bert_vector_chosen])

# GT和rejected的组合向量
combined_vector_gt_rejected_1 = np.concatenate([bow_vector_gt, bert_vector_gt])
combined_vector_gt_rejected_2 = np.concatenate([bow_vector_rejected, bert_vector_rejected])

# 计算组合向量的相似度
combined_similarity_gt_chosen = cosine_similarity([combined_vector_gt_chosen_1], [combined_vector_gt_chosen_2])[0][0]
combined_similarity_gt_rejected = cosine_similarity([combined_vector_gt_rejected_1], [combined_vector_gt_rejected_2])[0][0]

print(f"Combined Similarity (GT vs Chosen): {combined_similarity_gt_chosen:.4f}")
print(f"Combined Similarity (GT vs Rejected): {combined_similarity_gt_rejected:.4f}")
