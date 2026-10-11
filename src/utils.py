import numpy as np
from pathlib import Path

# import inspect
# from bmi_free.io import import_coor_selection, path, import_ecog_prep, import_onsets
# import mne

def shuffle(X, dim, rng):
    """对 X 的 dim 维做打乱，每个样本(axis 0)独立使用不同的打乱方式，返回结果和用到的索引。"""
    X = np.moveaxis(X, dim, 1)          # -> (N, size_d, ...)
    N, size_d = X.shape[0], X.shape[1]

    rand = rng.random((N, size_d))
    perm_idx = np.argsort(rand, axis=1)  # (N, size_d)，每个样本一套独立排列

    reshaped = perm_idx.reshape(N, size_d, *([1] * (X.ndim - 2)))
    X_shuffled = np.take_along_axis(X, reshaped, axis=1)

    return np.moveaxis(X_shuffled, 1, dim), perm_idx



def select_epoch(epoch_start, epoch_end, annot_start, annot_end, remove_annot=True):

    epoch_start = np.asarray(epoch_start)
    epoch_end = np.asarray(epoch_end)
    annot_start = np.asarray(annot_start)
    annot_end = np.asarray(annot_end)

    overlap = (
        epoch_start[:, None] <= annot_end[None, :]
    ) & (
        epoch_end[:, None] >= annot_start[None, :]
    )
    mask = overlap.any(axis=1)

    if remove_annot:
        mask = ~mask

    return np.where(mask)

def apply_transformation(cage_points, transformation_matrix):
    # 将cage_points转换为齐次坐标
    num_points = cage_points.shape[0]
    homogeneous_points = np.hstack(
        (cage_points, np.ones((num_points, 1))))  # shape (N, 4)

    # 应用变换矩阵
    transformed_homogeneous_points = (
        transformation_matrix @ homogeneous_points.T).T  # shape (N, 4)

    # 转换回3D坐标
    transformed_points = transformed_homogeneous_points[:, :3]

    return transformed_points

class Tag(dict):
    """
    Tag 是一个有序 dict，键值对序列化为 _key[value] 格式。
    
    用法：
        tag = Tag(ncomp=10, time_range=[-1, 0])
        str(tag)  # '_ncomp[10]_time_range[[-1, 0]]'
        Tag.from_str('_ncomp[10]_time_range[[-1, 0]]')  # Tag({'ncomp': 10, 'time_range': [-1, 0]})
    """
    def __str__(self):
        return ''.join(f'_{k}[{v!r}]' for k, v in self.items())

    def __repr__(self):
        return f'Tag({dict.__repr__(self)})'
    
    def __or__(self, other):
        return Tag({**self, **other})

    @classmethod
    def from_str(cls, tag_str: str) -> 'Tag':
        import re, ast
        pattern = re.compile(r'_([a-zA-Z][a-zA-Z0-9_]*)\[(.+?)\](?=_[a-zA-Z]|$)')
        result = cls()
        for m in pattern.finditer(tag_str):
            key, val_str = m.group(1), m.group(2)
            try:
                result[key] = ast.literal_eval(val_str)
            except (ValueError, SyntaxError):
                result[key] = val_str
        return result
    
def simp_idx(nums):
    nums = sorted(set(nums))
    groups = []
    start = end = nums[0]
    for n in nums[1:]:
        if n == end + 1:
            end = n
        else:
            groups.append(f'{start}-{end}' if start != end else str(start))
            start = end = n
    groups.append(f'{start}-{end}' if start != end else str(start))
    return '[' + ', '.join(groups) + ']'

def p2aster(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'n.s'



def p2aster(pval, thresholds='Regular'):
    if thresholds == 'Regular':
        map = {0.001: '***', 0.01: '**', 0.05: '*'}
    elif thresholds == 'Loose':
        map  = {0.001: '***', 0.01: '**', 0.05: '*', 0.1: '.'}  
    for threshold, symbol in map.items():
        if pval < threshold:
            return symbol
    return 'n.s'