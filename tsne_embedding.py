import numpy as np, matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE


video_token_hidden_states = torch.load("video_token_hidden_states.pt")

query_token_hidden_states = torch.load("query_token_hidden_states.pt")

video_token_hidden_states = torch.mean(video_token_hidden_states, dim=1)
query_token_hidden_states = torch.mean(query_token_hidden_states, dim=1)

tokens = torch.cat([video_token_hidden_states, query_token_hidden_states], dim=0)

print(video_token_hidden_states.shape[1])
print(query_token_hidden_states.shape[1])

colors = ['lightskyblue'] * video_token_hidden_states.shape[1] + ['green'] * query_token_hidden_states.shape[1]

array = tokens.numpy()

print(array.shape)

tsne = TSNE(n_components=2, perplexity=1, random_state=42)

mappings = tsne.fit_transform(array)

# 2D scatter
x, y = mappings[:, 0], mappings[:, 1]
plt.figure(figsize=(10, 7))
plt.scatter(x, y, label='Data Points')

plt.title('2D t-SNE Visualization', fontsize=16)
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)

import math
dis = math.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)

print(f"dis: {dis}")

output_path = "tsne_2d.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 高分辨率保存
print(f"图像已保存为 {output_path}")


# 3D scatter

# x, y, z = mappings[:, 0], mappings[:, 1], mappings[:, 2]

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(x, y, z, c=colors, marker='o', alpha=0.5)

# ax.set_title('3D t-SNE Visualization', fontsize=16)
# ax.set_xlabel('X', fontsize=12)
# ax.set_ylabel('Y', fontsize=12)
# ax.set_zlabel('Z', fontsize=12)
# # plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)

# output_path = "tsne_3d.png"
# plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 高分辨率保存
# print(f"图像已保存为 {output_path}")