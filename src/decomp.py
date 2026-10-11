import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
# import tensorly as tl
# from tensorly.decomposition import tucker
# from bmi_free.plot import palette
# from bmi_free.io import import_joints
from sklearn.decomposition import PCA


# def decompose(weights, method='tucker'):
#     if method == 'tucker':
#         return _tucker_als(weights)

# def show_joint_sketch(joints, conns, coords=None):
#     """
#     根据关节坐标绘制散点图，并画出关节之间的连接关系。

#     参数:
#         joints: 关节名称列表（与 coords 顺序一致）
#         conn: 关节连接关系列表，如 [['Head', 'Nose'], ...]
#         coords: 关节坐标列表，如 [(0, 0), (0, 1), ...]
#         point_size: 散点大小
#         figsize: 图像尺寸，默认根据坐标范围自动估算
#         save_path: 若提供，则保存图片到该路径
#     """
#     if coords is None:
#         coords = dict(zip(joints, [
#             (-1, 1), ( 0, 1), ( 0, 2),
#             (-1, 2), (-2, 2), (-3, 2),
#             ( 1, 2), ( 2, 2), ( 3, 2),
#             ( 0, 3), ( 0, 4),
#             (-1, 4), (-1, 5), (-1, 6),
#             ( 1, 4), ( 1, 5), ( 1, 6),
#         ]))

#     coords_arr = np.array([*coords.values()])
#     x_range = coords_arr[:, 0].max() - coords_arr[:, 0].min()
#     y_range = coords_arr[:, 1].max() - coords_arr[:, 1].min()

#     fig, ax = plt.subplots(figsize=(4, 4))

#     # 先画骨骼连接线（置于底层）
#     for j1, j2 in conns:
#         if j1 in joints and j2 in joints:
#             x1, y1 = coords[j1]
#             x2, y2 = coords[j2]
#             ax.plot([x1, x2], [y1, y2], color="k", linewidth=1,
#                     zorder=1, solid_capstyle="round")

#     # 画散点
#     xs = coords_arr[:, 0]
#     ys = coords_arr[:, 1]
#     ax.scatter(xs, ys, c=[palette['joint'][joint] for joint in joints],
#                zorder=2)

#     # 标注关节名称
#     for name, (x, y) in coords.items():
#         if name in ['Head', 'Nose', 'Neck', 'Back']:
#             ax.annotate(name.replace('(', '\n('), 
#                         (x+.1, y-.1), 
#                         va="baseline", ha="left",
#                         textcoords="offset points",
#                         xytext=(0, 0), fontsize=8)
#         elif x < 0 and y > 3:
#             ax.annotate(name, 
#                         (x-.15, y), 
#                         va="center", ha="right",
#                         textcoords="offset points",
#                         xytext=(0, 0), fontsize=8)   
#         elif x > 0 and y > 3:
#             ax.annotate(name, 
#                         (x+.15, y), 
#                         va="center", ha="left",
#                         textcoords="offset points",
#                         xytext=(0, 0), fontsize=8)                      
#         else:
#             ax.annotate(name.replace('(', '\n('), 
#                         (x, y+0.15), 
#                         va="top", ha="center",
#                         textcoords="offset points",
#                         xytext=(0, 0), fontsize=8)

#     ax.invert_yaxis()  # y 越大越靠下（脚），反转后图像呈现“头上脚下”的直觉方向
#     ax.set_aspect("equal")
#     ax.set_xlim(coords_arr[:, 0].min() - 1.2, coords_arr[:, 0].max() + 1.6)
#     ax.set_ylim(coords_arr[:, 1].max() + 0.6, coords_arr[:, 1].min() - 0.6)
#     ax.set_axis_off()

#     plt.tight_layout()
#     plt.show()


# def _tucker_als(weights, R_freq=2, R_chan=2):
#     W = weights.get_data()
#     times = weights.times
#     joints = weights.metadata['joint']

#     n_joint, n_chan, n_freq, n_time = W.shape
#     n_sample = n_joint * n_time
 
#     # (joint, chan, freq, time) -> (freq, chan, joint, time) -> (freq, chan, joint*time)
#     W_reordered = np.transpose(W, (2, 1, 0, 3))
#     W_3way = W_reordered.reshape(n_freq, n_chan, n_sample)
 
#     core, factors = tucker(
#         tl.tensor(W_3way),
#         rank=[R_freq, R_chan, n_sample],
#     )
#     U_freq = tl.to_numpy(factors[0])   # (n_freq, R_freq)
#     U_chan = tl.to_numpy(factors[1])   # (n_chan, R_chan)
 
 
#     freq_pc1 = U_freq[:, 0]
#     chan_pc1 = U_chan[:, 0]
 
#     chan_proj = np.einsum('fcs,c->fs', W_3way, chan_pc1)     # (n_freq, n_sample)
#     chan_score_flat = chan_proj.mean(axis=0)                 # (n_sample,)
 
#     freq_proj = np.einsum('fcs,f->cs', W_3way, freq_pc1)     # (n_chan, n_sample)
#     freq_score_flat = freq_proj.mean(axis=0)                 # (n_sample,)
 
#     # reshape 回 (n_joint, n_time);因为合并时是 (..., n_joint, n_time) -> (..., n_joint*n_time)
#     # 按 C-order flatten 的,所以这里直接 reshape 回去顺序就是对的
#     chan_score = chan_score_flat.reshape(n_joint, n_time)
#     freq_score = freq_score_flat.reshape(n_joint, n_time)
    
#     col_idx = pd.MultiIndex.from_tuples(
#         [(j, dim) for j in joints for dim in ['freq', 'chan']],
#         names=['joint', 'dim'],
#     )

#     # Stack to (n_joint, 2, n_time) then reshape to (n_time, n_joint*2)
#     # axis=1 interleave order matches [(j,'freq'),(j,'chan'), ...]
#     data = np.stack([freq_score, chan_score], axis=1)  # (n_joint, 2, n_time)
#     data = data.reshape(n_joint * 2, n_time).T          # (n_time, n_joint*2)

#     return pd.DataFrame(data, index=times, columns=col_idx)    
 

# def viz_space(df, conns, cmap=None):
#     times = df.index.to_numpy()
#     n_time = len(times)

#     joints = df.columns.get_level_values('joint').unique().tolist()

#     fig, ax = plt.subplots(2, 5, figsize=(10, 4), sharey='all', sharex='all')
#     ax = ax.flatten()
    
#     if cmap is None:
#         cmap = list(palette['joint'].values())
    
#     for i, t in enumerate(times):
#         plt.axes(ax[i])

#         for conn in conns:
#             if conn[0] in joints and conn[1] in joints:
#                 ax[i].plot(
#                     df.loc[t, (conn, 'freq')], df.loc[t, (conn, 'chan')],
#                     color="k", linewidth=1, zorder=1, solid_capstyle="round",
#                 )

#         plt.scatter(df.loc[t, (slice(None), 'freq')], df.loc[t, (slice(None), 'chan')], color=cmap, s=10)
#         plt.xticks([])
#         plt.yticks([])
#         plt.title(f"{t:.1f} ~ {t+0.1:.1f} s")

#     plt.tight_layout()
#     plt.show()
    

# def compute_skeleton_res():
#     df_pca_weights_list = []
#     weights_list = []
#     df_pc = pd.DataFrame(index=pd.MultiIndex.from_product([sessions, path['results'], joints], 
#                                                         names=['Session', 'Group', 'Joint']), columns=['PC0', 'PC1', 'PC2'])
#     df_contrib = {'chan': [], 'freq': []}
#     for p in path['results']:
#         weights = load_weights(p, absolute=False, baseline=(None, None))
#         freqs = weights[sessions[0]].freqs
#         times = weights[sessions[0]].times

#         for session in sessions:
            
#             W = weights[session].get_data()
#             n_joints, n_chans, n_freqs, n_times = W.shape
#             freqs = weights[session].freqs
#             times = weights[session].times
#             chans = weights[session].ch_names
#             # joints = weights[session].metadata['joint'].tolist()
#             W = W.transpose([0, 3, 1, 2]).reshape(-1, n_chans * n_freqs * n_times)

#             # 初始化 PCA，n_components 指定保留的维度数
#             pca = PCA(n_components=10, random_state=98)
#             pca.fit(W)

#             # PCA 权重矩阵 (feat, n_components)
#             pca_weights = pca.components_.T        # shape: (feat, n_components)
#             df_pca_weights = pd.DataFrame(pca_weights, index=pd.MultiIndex.from_product([chans, freqs, times], names=['Channel', 'Frequency', 'Time']), columns=range(pca.n_components))
#             df_pca_weights = df_pca_weights.reset_index().melt(id_vars=['Channel', 'Frequency', 'Time'], var_name='PC', value_name='Weight')
#             df_pca_weights['Monkey'] = session[0]
#             df_pca_weights_list.append(df_pca_weights)

#             # 各主成分解释的方差比例
#             var = pca.explained_variance_ratio_    # shape: (n_components,)

#             # 如果需要将 W 投影到低维空间
#             W_reduced = pca.transform(W)           # shape: (sample, n_components)
#             W_reduced = W_reduced[:, :3].reshape(n_joints, -1)
#             # df_pc.loc[(session, p.name, joints), :] = pd.DataFrame(W_reduced, index=joints, columns=[f'PC{i}' for i in range(3)])
#             df_pc.loc[(session, p, joints), :] = W_reduced

#             # 绘制解释方差比例
#             cu_var = np.cumsum(var)
#             plt.plot(range(pca.n_components+1), [0]+cu_var.tolist(), marker='o', color=palette['monkey'][session[0]], label=session[0])
#             plt.xticks(range(0, pca.n_components, 2))
#             plt.ylabel('Explained variance')
#             plt.xlabel('Principal component')
#             plt.legend()
            
#         plt.show()
def decompose_neural_weights(weights):
    freqs = weights.freqs
    times = weights.times
    joints = weights.metadata['joint'].tolist()

    W = weights.get_data()
    n_joints, n_chans, n_freqs, n_times = W.shape
    W = W.transpose([0, 3, 1, 2]).reshape(-1, n_chans * n_freqs * n_times)

    # 初始化 PCA，n_components 指定保留的维度数
    pca = PCA(n_components=10, random_state=98)
    pca.fit(W)

    # 如果需要将 W 投影到低维空间
    W_reduced = pca.transform(W)           # shape: (sample, n_components)
    W_reduced = W_reduced[:, :3].reshape(n_joints, -1)
    df_pc = pd.DataFrame(W_reduced, index=pd.Categorical(joints, ordered=True), columns=['PC0', 'PC1', 'PC2'])    
    return df_pc