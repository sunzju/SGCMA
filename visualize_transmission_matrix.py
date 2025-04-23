import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


check = torch.load("checkpoints/long_term_forecast_ETTh1_ftM_sl96_pl96_lr0.0001_lradjPEMS_0_cmt1118/checkpoint", weights_only=True)

transition_logits_ts = check["ts_cluster.ts_transition_logits"]

transition_logits_txt = torch.load("hmm/transition_logits.pt", weights_only=False)
transition_matrix_txt = torch.softmax(transition_logits_txt, dim=1).cpu().detach().numpy() # Convert to probability matrix

threshold = 0.05

transition_matrix_ts = torch.softmax(transition_logits_ts, dim=1).cpu().detach().numpy() # 转换为概率矩阵

# Create directed graph
G = nx.DiGraph()
for i in range(transition_matrix_ts.shape[0]):
    for j in range(transition_matrix_ts.shape[1]):
        if transition_matrix_ts[i, j] > threshold and transition_matrix_txt[i, j] >threshold: # Only draw edges with probability greater than 0.1
            G.add_edge(i, j, weight=transition_matrix_ts[i, j], color='blue')
        elif transition_matrix_ts[i, j] > threshold:
            G.add_edge(i, j, weight=transition_matrix_ts[i, j], color='red')
        elif transition_matrix_txt[i, j] > threshold:
            G.add_edge(i, j, weight=transition_matrix_txt[i, j], color='green')



print(np.linalg.norm(transition_matrix_txt - transition_matrix_ts,1))



# 绘制图
plt.figure(figsize=(12, 10))
pos = nx.random_layout(G)
edges = G.edges(data=True)
colors = [edge[2]['color'] for edge in edges]
nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", 
    edge_color=colors, width=1.5, font_size=10)
plt.title("State Transition Graph")
plt.savefig("transition_graph.png", bbox_inches="tight")
plt.close()