import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm


data_path="/volsparse1/wxd/data/shareVideoGPTV/ov-72b-f32_next-7b-DPO-iter1-sample-K1_0_8000.jsonl"

data = []
with open(data_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        entry = json.loads(line)
        data.append(entry)

device = "cuda:0"
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
model = BertModel.from_pretrained('google-bert/bert-base-uncased').to(device)

def infer(gt, chosen, rejected):

    inputs_gt = tokenizer(gt, return_tensors='pt', max_length=128, truncation=True, padding=True).to(device)
    inputs_chosen = tokenizer(chosen, return_tensors='pt', max_length=128, truncation=True, padding=True).to(device)
    inputs_rejected = tokenizer(rejected, return_tensors='pt', max_length=128, truncation=True, padding=True).to(device)


    with torch.no_grad():
        outputs_gt = model(**inputs_gt)
        outputs_chosen = model(**inputs_chosen)
        outputs_rejected = model(**inputs_rejected)


    gt_embedding = outputs_gt.last_hidden_state[:, 0, :].cpu().numpy()
    chosen_embedding = outputs_chosen.last_hidden_state[:, 0, :].cpu().numpy()
    rejected_embedding = outputs_rejected.last_hidden_state[:, 0, :].cpu().numpy()


    bert_similarity_gt_chosen = cosine_similarity(gt_embedding, chosen_embedding)[0][0]
    bert_similarity_gt_rejected = cosine_similarity(gt_embedding, rejected_embedding)[0][0]

    # print(f"BERT similarity (GT vs Chosen): {bert_similarity_gt_chosen:.4f}")
    # print(f"BERT similarity (GT vs Rejected): {bert_similarity_gt_rejected:.4f}")


    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform([gt, chosen, rejected])


    similarity_matrix = cosine_similarity(X)

    wordbag_similarity_gt_chosen = similarity_matrix[0][1]
    wordbag_similarity_gt_rejected = similarity_matrix[0][2]

    # print(f"WordBag similarity (GT vs Chosen): {wordbag_similarity_gt_chosen:.4f}")
    # print(f"WordBag similarity (GT vs Rejected): {wordbag_similarity_gt_rejected:.4f}")


    bow_vector_gt = X.toarray()[0]
    bert_vector_gt = gt_embedding.flatten()

    bow_vector_chosen = X.toarray()[1]
    bert_vector_chosen = chosen_embedding.flatten()

    bow_vector_rejected = X.toarray()[2]
    bert_vector_rejected = rejected_embedding.flatten()


    combined_vector_gt_chosen_1 = np.concatenate([bow_vector_gt, bert_vector_gt])
    combined_vector_gt_chosen_2 = np.concatenate([bow_vector_chosen, bert_vector_chosen])


    combined_vector_gt_rejected_1 = np.concatenate([bow_vector_gt, bert_vector_gt])
    combined_vector_gt_rejected_2 = np.concatenate([bow_vector_rejected, bert_vector_rejected])

    combined_similarity_gt_chosen = cosine_similarity([combined_vector_gt_chosen_1], [combined_vector_gt_chosen_2])[0][0]
    combined_similarity_gt_rejected = cosine_similarity([combined_vector_gt_rejected_1], [combined_vector_gt_rejected_2])[0][0]

    return combined_similarity_gt_chosen, combined_similarity_gt_rejected
    # print(f"Combined Similarity (GT vs Chosen): {combined_similarity_gt_chosen:.4f}")
    # print(f"Combined Similarity (GT vs Rejected): {combined_similarity_gt_rejected:.4f}")


for idx, item in tqdm(enumerate(data)):
    chosen_score, rejected_score = infer(item["answer"], item["chosen"], item["rejected"])
    data[idx]["chosen_bert_score"] = chosen_score
    data[idx]["rejected_bert_score"] = rejected_score

output_file="/volsparse1/wxd/data/shareVideoGPTV/ov-72b-f32_next-7b-DPO-iter1-sample-K1_0_8000_bert.jsonl"
with open(output_file, 'w', encoding='utf-8') as outfile:
    for line in data:
        json.dump(line, outfile)  # Write the JSON entry to the file
        outfile.write('\n')  # Add a newline after each entry

print(data[0])
print("completed!")