import numpy as np
from pathlib import Path

# import inspect
# from bmi_free.io import import_coor_selection, path, import_ecog_prep, import_onsets
# import mne

def shuffle(X, dim, rng):
    """
    对 X 的指定维度做打乱，每个样本 (axis 0) 独立使用不同的打乱方式。

    Parameters
    ----------
    X : numpy.ndarray
        输入数组。
    dim : int
        要打乱的维度索引。
    rng : numpy.random.Generator
        随机数生成器。

    Returns
    -------
    X_shuffled : numpy.ndarray
        打乱后的数组。
    perm_idx : numpy.ndarray
        使用的排列索引，shape (N, size_d)。
    """
    X = np.moveaxis(X, dim, 1)          # -> (N, size_d, ...)
    N, size_d = X.shape[0], X.shape[1]

    rand = rng.random((N, size_d))
    perm_idx = np.argsort(rand, axis=1)  # (N, size_d)，每个样本一套独立排列

    reshaped = perm_idx.reshape(N, size_d, *([1] * (X.ndim - 2)))
    X_shuffled = np.take_along_axis(X, reshaped, axis=1)

    return np.moveaxis(X_shuffled, 1, dim), perm_idx



def select_epoch(epoch_start, epoch_end, annot_start, annot_end, remove_annot=True):
    """
    根据标注区间筛选 epoch，返回与标注无重叠（或有重叠）的 epoch 索引。

    Parameters
    ----------
    epoch_start : array-like
        各 epoch 的起始时间。
    epoch_end : array-like
        各 epoch 的结束时间。
    annot_start : array-like
        各标注的起始时间。
    annot_end : array-like
        各标注的结束时间。
    remove_annot : bool
        若 True，返回与标注无重叠的 epoch；若 False，返回与标注有重叠的 epoch。

    Returns
    -------
    tuple of numpy.ndarray
        符合条件的 epoch 索引（元组格式，可直接用于 numpy 索引）。
    """

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
    """
    对 3D 点集施加齐次变换矩阵。

    Parameters
    ----------
    cage_points : numpy.ndarray, shape (N, 3)
        待变换的 3D 点坐标。
    transformation_matrix : numpy.ndarray, shape (4, 4)
        齐次变换矩阵。

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        变换后的 3D 点坐标。
    """
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
    """
    将整数列表压缩为简洁的区间表示字符串。

    Parameters
    ----------
    nums : iterable of int
        整数列表。

    Returns
    -------
    str
        压缩后的区间表示，如 '[1-3, 5, 7-9]'。

    Examples
    --------
    >>> simp_idx([1, 2, 3, 5, 7, 8, 9])
    '[1-3, 5, 7-9]'
    """
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

def p2aster(pval, thresholds='Regular'):
    """
    将 p 值转换为显著性星号标记。

    Parameters
    ----------
    pval : float
        p 值。
    thresholds : {'Regular', 'Loose'}
        显著性阈值方案：
        - 'Regular' : 0.001 → '***', 0.01 → '**', 0.05 → '*'
        - 'Loose' : 额外包含 0.1 → '.'

    Returns
    -------
    str
        显著性标记，如 '***'、'**'、'*'、'.' 或 'n.s'。
    """
    if thresholds == 'Regular':
        map = {0.001: '***', 0.01: '**', 0.05: '*'}
    elif thresholds == 'Loose':
        map  = {0.001: '***', 0.01: '**', 0.05: '*', 0.1: '.'}  
    for threshold, symbol in map.items():
        if pval < threshold:
            return symbol
    return 'n.s'