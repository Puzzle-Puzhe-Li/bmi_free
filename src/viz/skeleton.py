"""
skeleton.py
-----------
骨架可视化模块。通过 MATLAB Engine API for Python 调用本目录下 skeleton/
子目录中的 plot_skeleton.m，画风与 viz_kinetics.m 保持一致。

只有两个公开接口:
    - plot_skeleton_grid(coor, ...)      单帧 / 多帧网格子图（单帧是特例）
    - plot_skeleton_animation(coor, ...) 多帧导出视频

依赖:
    1. 本机已安装 MATLAB。
    2. 安装 MATLAB Engine API for Python，二选一:
         a) pip install matlabengine   （需与本地 MATLAB 版本匹配）
         b) cd "<matlabroot>/extern/engines/python" && python setup.py install
    3. 本文件所在目录下需要有 skeleton/ 子目录，里面放置随本文件一并
       提供的 plot_skeleton.m（'grid'/'animation' 两个功能合并在这一个
       文件里，通过第一个参数 mode 分发，因为 MATLAB 的函数文件只有
       第一个函数能被外部直接调用）。

输入约定:
    coor : pd.DataFrame
        - index : 时间（帧序号或时间戳）
        - columns : pd.MultiIndex，level-0 = joint name，level-1 = dim（'x','y','z' 或其他名称）
        - 取前3个 dim 作为空间坐标（不要求 dim 必须叫 xyz）

skeleton 连接关系由 import_skeleton_config() 读取，
内部自动从 edges 中检测三角面（三条边构成闭合三角的情况），
不再需要手动定义 tets。

plot_skeleton_grid 的 indices 参数决定绘制形态（对应"矩阵维度"）:
    - 单个 int / label（非 list）      : 单帧 —— 一个子图（1x1 网格）
    - 扁平 list[int|label] 或 None    : 多帧 —— 自动或按 shape 排成网格
    - 嵌套 list[list[int|label]]      : 多帧 —— 直接按嵌套结构指定网格排布
                                        （每个子 list 是网格的一行，各行
                                        长度需一致），忽略 shape 参数

笼子组件 (show_floor / show_poles / show_walls):
    默认均为 False。打开后会画出训练笼的地面网格 / 爬杆 / 墙壁，
    坐标取自 viz_kinetics.m 中被注释掉的对应代码块，是固定的实验装置尺寸。

视角 (view):
    None（默认）: MATLAB 用 (150, 10)（即原脚本默认视角）。
    (az, el)    : 沿用 MATLAB view(az, el) 的定义
                  （az: 绕 Z 轴方位角；el: 相对 XY 平面仰角）。
"""

import itertools
import os
import math
import tempfile

import numpy as np
import pandas as pd

from itertools import combinations
import networkx as nx

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr, wilcoxon

from bmi_free.io import import_skeleton_config, import_pc


# =========================================================
# 内部工具：坐标 / 连接关系
# =========================================================

def _coor_row_to_V(row):
    """
    把 coor 的单行（pd.Series，MultiIndex index）转成 [N_joint, 3] 的 numpy 数组。
    取每个 joint 的前3个 dim 作为空间坐标，dim 名称不限。
    """
    if isinstance(row, pd.DataFrame):
        return row.values[:, :3].astype(float)    
    
    joints = row.index.get_level_values(0).unique()
    coords = []
    for joint in joints:
        vals = row[joint].values
        coords.append(vals[:3].astype(float))
    return np.stack(coords, axis=0)  # [N_joint, 3]

def _row_joints(row):
    """
    从 coor 行中提取关节名称列表。

    Parameters
    ----------
    row : pd.Series or pd.DataFrame
        单行或多行坐标数据。

    Returns
    -------
    list of str
        关节名称列表。
    """
    if isinstance(row, pd.DataFrame):
        return row.index.tolist()
    return row.index.get_level_values(0).unique().tolist()

def _names_to_indices(name_pairs, joints, verbose=False):
    """
    把 [[nameA, nameB], ...] 转成 0-based 整数索引矩阵。
    任意一个 joint 不在 joints 里则跳过该条。
    """
    joint_index = {name: i for i, name in enumerate(joints)}
    result = []
    for pair in name_pairs:
        missing = [n for n in pair if n not in joint_index]
        if missing:
            if verbose:
                print(f"跳过 {pair}，缺失 joint: {missing}")
        else:
            result.append([joint_index[n] for n in pair])
    if not result:
        k = len(name_pairs[0]) if name_pairs else 2
        return np.empty((0, k), dtype=int)
    return np.array(result, dtype=int)


def _find_triangles(edge_idx):
    """
    从 edge 索引 [N, 2] 中自动检测所有三角面（三条边两两相连构成闭合三角）。
    返回 [M, 3] 的三角面索引。
    """
    edge_set = set(map(tuple, np.sort(edge_idx, axis=1).tolist()))
    nodes = np.unique(edge_idx)
    triangles = []
    for a, b, c in itertools.combinations(nodes, 3):
        if (min(a, b), max(a, b)) in edge_set and \
           (min(b, c), max(b, c)) in edge_set and \
           (min(a, c), max(a, c)) in edge_set:
            triangles.append([a, b, c])
    return np.array(triangles, dtype=int) if triangles else np.empty((0, 3), dtype=int)


# =========================================================
# 内部工具：bounds 计算
# =========================================================

def _pad_bounds(bounds, pad=0.2):
    """
    将 bounds 向四周扩展，避免点球体和线宽在视口边缘溢出。

    pad : float
        相对于各轴跨度的扩展比例（默认 20%）。
        若某轴跨度为 0（退化情况），则用其他轴平均跨度的 pad 倍兜底。
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    spans = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)
    nonzero = spans[spans > 0]
    fallback = float(nonzero.mean()) * pad if len(nonzero) else 1.0
    dx, dy, dz = [s * pad if s > 0 else fallback for s in spans]
    return [xmin - dx, xmax + dx, ymin - dy, ymax + dy, zmin - dz, zmax + dz]


def _compute_bounds(Vs):
    """
    给定顶点数组列表 [N_joint, 3]，返回各帧的 bounds 以及全局 bounds。

    返回
    ----
    per_bounds : list of [xmin, xmax, ymin, ymax, zmin, zmax]
    global_bounds : [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    per_bounds = []
    for V in Vs:
        mins = V.min(axis=0)
        maxs = V.max(axis=0)
        per_bounds.append([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]])

    arr = np.array(per_bounds)  # [n, 6]
    global_bounds = [
        arr[:, 0].min(), arr[:, 1].max(),   # x
        arr[:, 2].min(), arr[:, 3].max(),   # y
        arr[:, 4].min(), arr[:, 5].max(),   # z
    ]
    return per_bounds, global_bounds


def _parse_bounds(bounds, Vs, bound_scale='*1.1', joints=None):
    """
    解析 bounds 参数，返回统一的 [xmin, xmax, ymin, ymax, zmin, zmax] 格式。

    参数
    ----
    bounds : str | list
        - 'auto' : 自动根据数据计算范围
        - 'cage' : 使用笼子固定范围 [[0,70], [0,100], [0,100]]
        - '{joint}_center' : 以指定关节为中心（该关节在所有帧上的平均位置），
                              三轴各自向外扩展，scale=1.0 时恰好包住所有帧
                              所有关节点。例如 'Sacrum_center'。
        - [[xmin,xmax], [ymin,ymax], [zmin,zmax]] : 3x2 矩阵
        - [min, max] : 1x2 矩阵，三轴共用
    Vs : list of [N_joint, 3]
        所有帧的顶点数据
    bound_scale : str
        范围缩放表达式，如 '+5', '-5', '*1.5'
    joints : list[str] | None
        关节名称列表，与 Vs 里每帧的行顺序一一对应；使用 '{joint}_center'
        语法时必须提供。

    返回
    ----
    bounds : [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    if isinstance(bounds, str):
        if bounds == 'auto':
            all_V = np.vstack(Vs) if Vs else None
            if all_V is None or len(all_V) == 0:
                raise ValueError("没有数据可计算 auto bounds")
            mins = all_V.min(axis=0)
            maxs = all_V.max(axis=0)
            raw_bounds = [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]
        elif bounds == 'cage':
            raw_bounds = [0, 70, 0, 100, 0, 100]
        elif bounds.endswith('_center'):
            joint_name = bounds[:-len('_center')]
            if joints is None:
                raise ValueError(
                    f"使用 '{bounds}' 这种 '{{joint}}_center' 语法时必须提供 joints 列表"
                )
            if joint_name not in joints:
                raise ValueError(f"未找到关节 {joint_name!r}，可选: {joints}")
            j_idx = joints.index(joint_name)

            all_V = np.vstack(Vs) if Vs else None
            if all_V is None or len(all_V) == 0:
                raise ValueError(f"没有数据可计算 '{bounds}'")

            # 该关节在所有帧上的平均位置作为中心（单帧时就是它自己的位置）
            center_pos = np.mean([V[j_idx] for V in Vs], axis=0)
            # 所有帧、所有关节的点，相对这个中心在各轴上的最大绝对偏移，
            # 保证 scale=1.0 时三轴都恰好包住全部点
            half_range = np.max(np.abs(all_V - center_pos), axis=0)

            raw_bounds = [
                center_pos[0] - half_range[0], center_pos[0] + half_range[0],
                center_pos[1] - half_range[1], center_pos[1] + half_range[1],
                center_pos[2] - half_range[2], center_pos[2] + half_range[2],
            ]
        else:
            raise ValueError(f"未知 bounds 模式: {bounds!r}")
    elif isinstance(bounds, (list, tuple, np.ndarray)):
        bounds_arr = np.array(bounds).flatten()
        if len(bounds_arr) == 2:
            raw_bounds = [bounds_arr[0], bounds_arr[1]] * 3
        elif len(bounds_arr) == 6:
            raw_bounds = bounds_arr.tolist()
        else:
            raise ValueError(f"bounds 长度必须为 2 或 6，得到 {len(bounds_arr)}")
    else:
        raise ValueError(f"bounds 类型错误: {type(bounds)}")

    center = [
        (raw_bounds[0] + raw_bounds[1]) / 2,
        (raw_bounds[2] + raw_bounds[3]) / 2,
        (raw_bounds[4] + raw_bounds[5]) / 2,
    ]
    half_ranges = [
        (raw_bounds[1] - raw_bounds[0]) / 2,
        (raw_bounds[3] - raw_bounds[2]) / 2,
        (raw_bounds[5] - raw_bounds[4]) / 2,
    ]

    scale_str = bound_scale.strip()
    if scale_str.startswith('+') or scale_str.startswith('-'):
        delta = float(scale_str)
        new_half_ranges = [h + delta for h in half_ranges]
    elif scale_str.startswith('*'):
        factor = float(scale_str[1:])
        new_half_ranges = [h * factor for h in half_ranges]
    else:
        raise ValueError(f"bound_scale 格式错误: {bound_scale!r}，支持 '+5', '-5', '*1.5'")

    new_half_ranges = [max(0.1, h) for h in new_half_ranges]

    return [
        center[0] - new_half_ranges[0], center[0] + new_half_ranges[0],
        center[1] - new_half_ranges[1], center[1] + new_half_ranges[1],
        center[2] - new_half_ranges[2], center[2] + new_half_ranges[2],
    ]


# =========================================================
# 内部工具：帧选择 / 网格排布
# =========================================================

def _select_frames(coor, indices):
    """
    根据 indices（int 位置或 label 的扁平 list，或 None）从 coor 里选出若干行。
    返回 (rows, row_labels)。
    """
    if indices is None:
        rows = [coor.iloc[i] for i in range(len(coor))]
        row_labels = list(coor.index)
    else:
        rows, row_labels = [], []
        for idx in indices:
            if isinstance(idx, (int, np.integer)):
                rows.append(coor.iloc[idx])
                row_labels.append(coor.index[idx])
            else:
                rows.append(coor.loc[idx])
                row_labels.append(idx)
    return rows, row_labels


def _grid_shape(n, shape):
    """确定网格 (n_rows, n_cols)，None 时自动按接近正方形排布。"""
    if shape is None:
        n_cols = math.ceil(math.sqrt(n))
        n_rows = math.ceil(n / n_cols)
    else:
        n_rows, n_cols = shape
        if n_rows * n_cols < n:
            raise ValueError(
                f"shape=({n_rows},{n_cols}) 共 {n_rows*n_cols} 格，"
                f"不够放 {n} 帧，请调大 shape。"
            )
    return n_rows, n_cols


def _is_nested(indices):
    """indices 是否是"嵌套 list"（每个元素本身又是一个 list/tuple/ndarray），
    代表调用方直接指定了网格排布。"""
    if indices is None:
        return False
    if isinstance(indices, np.ndarray):
        return indices.ndim == 2
    return (
        isinstance(indices, (list, tuple))
        and len(indices) > 0
        and isinstance(indices[0], (list, tuple, np.ndarray))
    )


def _normalize_frame_selection(coor, indices, shape):
    """
    根据 indices 的结构自动判断绘制形态，返回 (rows, labels, n_rows, n_cols, is_single)。

    - indices 是单个 int/label（不是 list/tuple/ndarray）：
        单帧，1x1 网格（对应"2维矩阵"语义：一帧的 V 是 N x 3）。
    - indices 是扁平 list（或 None，取全部行）：
        多帧，按 shape 或自动排成网格（对应"3维矩阵"语义：V_all 是 T x N x 3）。
    - indices 是嵌套 list[list[...]]：
        多帧，直接按嵌套结构指定网格排布，每个子 list 是网格的一行，
        各行长度需一致（对应"4维矩阵"语义：网格是 n_rows x n_cols x N x 3），
        忽略 shape 参数。
    """
    if indices is not None and not isinstance(indices, (list, tuple, np.ndarray)):
        rows, labels = _select_frames(coor, [indices])
        return rows, labels, 1, 1, True

    if _is_nested(indices):
        row_lens = {len(r) for r in indices}
        if len(row_lens) != 1:
            raise ValueError(f"嵌套 indices 每行长度需一致，得到长度集合: {row_lens}")
        n_rows = len(indices)
        n_cols = row_lens.pop()
        flat_indices = [idx for row in indices for idx in row]
        rows, labels = _select_frames(coor, flat_indices)
        return rows, labels, n_rows, n_cols, False

    rows, labels = _select_frames(coor, indices)
    n = len(rows)
    if n == 0:
        raise ValueError("没有可绘制的帧，请检查 indices 或 coor。")
    if n == 1:
        return rows, labels, 1, 1, True
    n_rows, n_cols = _grid_shape(n, shape)
    return rows, labels, n_rows, n_cols, False


# =========================================================
# MATLAB engine 管理 / 导出显示
# =========================================================

# plot_skeleton.m 所在目录（与本文件同级）
_MATLAB_SKELETON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skeleton")

# MATLAB engine 单例：懒启动，进程内复用，避免每次调用都重新起 MATLAB（很慢）
_MATLAB_ENGINE = None


def _get_matlab_engine():
    """懒启动并返回全局唯一的 MATLAB engine 实例。"""
    global _MATLAB_ENGINE
    if _MATLAB_ENGINE is None:
        try:
            import matlab.engine  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "需要 MATLAB Engine API for Python。\n"
                "安装方法（二选一）：\n"
                "  1) pip install matlabengine   (需与本地 MATLAB 版本匹配)\n"
                "  2) cd \"<matlabroot>/extern/engines/python\" && python setup.py install"
            ) from e
        import matlab.engine
        print("正在启动 MATLAB engine（首次调用可能需要 10~20 秒）...")
        _MATLAB_ENGINE = matlab.engine.start_matlab()
        _MATLAB_ENGINE.addpath(_MATLAB_SKELETON_DIR, nargout=0)
    return _MATLAB_ENGINE


def close_matlab_engine():
    """手动关闭并释放当前 MATLAB engine 会话（如需彻底结束 MATLAB 进程时调用）。"""
    global _MATLAB_ENGINE
    if _MATLAB_ENGINE is not None:
        try:
            _MATLAB_ENGINE.quit()
        finally:
            _MATLAB_ENGINE = None


def _display_matlab_figure(eng, fig_handle, img_format="png", dpi=150):
    """
    把 MATLAB 图窗导出为临时图片文件，在 notebook 中内联显示后删除该临时文件。
    不影响 save_as 指定的 .fig 文件（那是另一份、会持久保留）。
    """
    from IPython.display import display, Image, SVG

    suffix = ".png" if img_format == "png" else ".svg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    tmp.close()  # 先关闭文件句柄，避免 Windows 下 MATLAB 写入时的文件占用冲突

    try:
        if img_format == "png":
            eng.exportgraphics(fig_handle, tmp_path, "Resolution", float(dpi), nargout=0)
            display(Image(filename=tmp_path))
        elif img_format == "svg":
            eng.exportgraphics(fig_handle, tmp_path, "ContentType", "vector", nargout=0)
            display(SVG(filename=tmp_path))
        else:
            raise ValueError(f"未知 img_format: {img_format!r}，可选 'png' 或 'svg'")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _to_matlab_view(view):
    """
    将 Python view 列表转换为 MATLAB double 数组。

    Parameters
    ----------
    view : list or None
        视角参数列表。

    Returns
    -------
    matlab.double
        MATLAB 兼容的 double 数组。
    """
    import matlab
    return matlab.double(list(view)) if view is not None else matlab.double([])


def _to_matlab_idx(idx_arr):
    """
    将 0-based numpy 索引矩阵转换为 1-based MATLAB double 数组。

    Parameters
    ----------
    idx_arr : numpy.ndarray
        0-based 索引矩阵。

    Returns
    -------
    matlab.double
        1-based MATLAB 索引数组；空则返回 matlab.double([])。
    """
    import matlab
    return matlab.double((idx_arr + 1).tolist()) if len(idx_arr) else matlab.double([])


# =========================================================
# 高层接口
# =========================================================

def plot_skeleton_grid(
    coor,
    indices=None,
    shape=None,
    edges=None,
    titles=None,
    bounds='auto',
    bound_scale='*1.1',
    save_as=None,
    show=True,
    # visible=True,
    # view=None,
    # show_floor=False,
    # show_poles=False,
    # show_walls=False,        
    # show_axis=False,
    # face_color='#9a7350',
    # face_alpha=0.7,
    # edge_color='k',
    # line_width=1,
    img_format="png",
    dpi=150,
    subplot_size=(400, 400),
    **kwargs,
):
    """
    绘制骨架图：单帧或多帧网格子图，单帧只是"网格恰好是 1x1"的特例。

    参数
    ----
    coor : pd.DataFrame
        骨架坐标表，index = 帧（行），columns = MultiIndex(joint, dim)。
    indices : int | label | list | list[list] | None
        决定绘制形态：
        - 单个 int / label            : 单帧（一个子图）
        - 扁平 list（或 None=全部行）  : 多帧，按 shape 或自动排成网格
        - 嵌套 list[list[...]]        : 多帧，直接按嵌套结构指定网格排布
          （每个子 list 是网格的一行，各行长度需一致），此时忽略 shape
    shape : (int, int) | None
        扁平多帧场景下的子图网格 (n_rows, n_cols)；None 则自动按接近
        正方形排布。对嵌套 indices 无效（网格形状由嵌套结构决定）。
    edges : list | None
        骨架连接关系；None 则自动调用 import_skeleton_config()。
    titles : list[str] | None
        每个子图的标题列表，长度需与帧数一致；None 时多帧用帧 index 值
        作标题，单帧不加标题。
    bounds : str | list, default='auto'
        坐标轴范围控制：
        - 'auto' : 自动根据数据范围计算
        - 'cage' : 使用笼子固定范围 [[0,70], [0,100], [0,100]]
        - [[xmin,xmax], [ymin,ymax], [zmin,zmax]] : 3x2 矩阵分别指定
        - [min, max] : 1x2 矩阵，三轴共用此范围
        所有子图共用同一个（padding 后的）范围。
    bound_scale : str, default='*1.1'
        范围缩放表达式，以当前范围中心为原点：'+5' / '-5' / '*1.5'。
    save_as : str | None
        指定则把图窗保存为 .fig 文件（可在 MATLAB 里重新打开、继续交互式
        旋转/编辑）；默认 None，不保存。
    show : bool
        默认 True。把图窗导出为一张临时图片，在 notebook 里内联显示后
        立即删除该临时图片文件（不影响 save_as 保留的 .fig）。
    visible : bool
        默认 True，弹出一个真实的 MATLAB 图窗（可交互旋转）。若只想要
        notebook 内联显示、不想每次都多开一个 MATLAB 窗口，可设为 False。
    view : (az, el) | None
        相机角度，沿用 MATLAB view(az,el) 的定义；None 用默认 (150, 10)。
    show_floor / show_poles / show_walls / show_axis : bool
        是否画地面网格 / 训练笼爬杆 / 墙壁网格与标签 / 坐标轴刻度，默认 False。
    img_format : {'png', 'svg'}
        show=True 时内联显示用的图片格式，默认 'png'。
    dpi : int
        img_format='png' 时的导出分辨率，默认 150。
    subplot_size : (int, int)
        多子图时，每个子图的像素尺寸 (width, height)，用于设置图窗大小。

    返回
    ----
    fig_handle : MATLAB 图窗句柄，可继续传给 eng.savefig 等

    示例
    ----
    >>> plot_skeleton_grid(coor, indices=46910)                       # 单帧
    >>> plot_skeleton_grid(coor, indices=[0, 10, 20, 30], shape=(2, 2))
    >>> plot_skeleton_grid(coor, indices=[[0, 1, 2], [3, 4, 5]])       # 显式 2x3 网格
    >>> plot_skeleton_grid(coor.iloc[::50], show_floor=True)
    """

    # kwargs['show_poles'] = kwargs.get('show_poles', False)
    # kwargs['show_floor'] = kwargs.get('show_floor', False)
    # kwargs['show_walls'] = kwargs.get('show_walls', False)
    # kwargs['face_color'] = kwargs.get('face_color', '#9a7350')
    # kwargs['face_alpha'] = kwargs.get('face_alpha', 0.7)
    # kwargs['edge_color'] = kwargs.get('edge_color', 'k')
    # kwargs['line_width'] = kwargs.get('line_width', 1)
    # face_alpha=0.7,
    # edge_color='k',
    # line_width=1,

    import matlab

    if edges is None:
        edges = import_skeleton_config()

    if isinstance(coor, pd.DataFrame) and not isinstance(coor.columns, pd.MultiIndex):
        # coor 本身就是一帧的关节坐标（index=joint, columns=dim），如 df_pc[session]
        rows, labels, n_rows, n_cols, is_single = [coor], [None], 1, 1, True
    else:
        rows, labels, n_rows, n_cols, is_single = _normalize_frame_selection(coor, indices, shape)

    n = len(rows)

    if titles is None:
        titles = [] if is_single else [
            f"{t:.1f}" if isinstance(t, (int, float, np.floating)) else str(t)
            for t in labels
        ]
    elif len(titles) < n:
        raise ValueError(f"titles 长度 ({len(titles)}) 小于帧数 ({n})。")

    joints = _row_joints(rows[0])
    edge_idx = _names_to_indices(edges, joints)
    tri_idx = _find_triangles(edge_idx) if len(edge_idx) else np.empty((0, 3), dtype=int)

    Vs = [_coor_row_to_V(row) for row in rows]

    if isinstance(bounds, str) or isinstance(bounds, (list, tuple, np.ndarray)):
        global_bounds = _parse_bounds(bounds, Vs, bound_scale, joints=joints)
    else:
        _, global_bounds = _compute_bounds(Vs)
        global_bounds = _pad_bounds(global_bounds)
    bounds_list = [global_bounds for _ in Vs]

    eng = _get_matlab_engine()

    V_all = np.stack(Vs, axis=0)  # [T, N, 3]
    T, N, _ = V_all.shape
    # matlab.double 要求 size 搭配一个按列主序(Fortran order)排好的扁平序列
    V_all_m = matlab.double(V_all.flatten(order="F").tolist(), size=(T, N, 3))
    edges_m = _to_matlab_idx(edge_idx)
    faces_m = _to_matlab_idx(tri_idx)
    bounds_all_m = matlab.double(np.array(bounds_list).tolist())
    save_as_m = save_as if save_as else ""
    # view_m = _to_matlab_view(view)

    fig_handle = eng.plot_skeleton(
        "grid", V_all_m, edges_m, faces_m, bounds_all_m, titles,
        float(n_rows), float(n_cols),
        # "save_as", save_as_m,
        # "visible", bool(visible),
        # "show_floor", bool(show_floor),
        # "show_poles", bool(show_poles),
        # "show_walls", bool(show_walls),
        # "show_axis", bool(show_axis),
        # "view_az_el", view_m,
        kwargs,
        nargout=1,
    )

    if not is_single:
        w, h = subplot_size
        eng.set(fig_handle, "position", np.array([0, 0, w * n_cols, h * n_rows]), nargout=0)

    if show:
        _display_matlab_figure(eng, fig_handle, img_format=img_format, dpi=dpi)

    if save_as:
        eng.print(fig_handle, save_as, '-dpdf', '-vector', nargout=0)

    return eng


def plot_skeleton_animation(
    coor,
    edges=None,
    fps=30,
    filename="animation.mp4",
    bounds='auto',
    bound_scale='*1.1',
    visible=True,
    view=None,
    show_floor=False,
    show_poles=False,
    show_walls=False,
    show_axis=False,
):
    """
    渲染逐帧动画并导出为视频。

    一次性把全部帧传给 MATLAB，由 MATLAB 内部循环更新图形对象并写视频
    （与原 viz_kinetics.m 的 plotAnimation 逻辑一致），因此整段动画只需
    一次 MATLAB Engine 调用，效率较高。

    注意：帧数很大时（数万帧级别），一次性传输 + 写视频耗时会比较长，
    请预留时间，或先用 coor.iloc[a:b] 截取需要的片段再调用。

    参数
    ----
    coor : pd.DataFrame
        要渲染的帧序列（建议传入已截取好的片段，如 coor.iloc[2400:2700]）。
    edges : list | None
        骨架连接关系；None 则自动调用 import_skeleton_config()。
    fps : int
        帧率，默认 30。
    filename : str
        输出 mp4 文件路径。
    bounds : str | list, default='auto'
        坐标轴范围控制，含义同 plot_skeleton_grid；动画期间所有帧共用
        同一个范围（基于全部待渲染帧的数据算好，保持固定不随帧变化）。
    bound_scale : str, default='*1.1'
        范围缩放表达式，含义同 plot_skeleton_grid。
    visible, view, show_floor, show_poles, show_walls, show_axis :
        含义同 plot_skeleton_grid。

    示例
    ----
    >>> plot_skeleton_animation(coor.iloc[2400:2700], filename="demo.mp4")
    >>> plot_skeleton_animation(coor.iloc[2400:2700], show_floor=True, show_poles=True, view=(60, 15))
    """
    import matlab

    if edges is None:
        edges = import_skeleton_config()

    joints = coor.iloc[0].index.get_level_values(0).unique().tolist()
    edge_idx = _names_to_indices(edges, joints)
    tri_idx = _find_triangles(edge_idx) if len(edge_idx) else np.empty((0, 3), dtype=int)

    Vs = [_coor_row_to_V(coor.iloc[i]) for i in range(len(coor))]

    if isinstance(bounds, str) or isinstance(bounds, (list, tuple, np.ndarray)):
        global_bounds = _parse_bounds(bounds, Vs, bound_scale, joints=joints)
    else:
        _, parsed_bounds = _compute_bounds(Vs)
        parsed_bounds = _pad_bounds(parsed_bounds)

    eng = _get_matlab_engine()

    V_all = np.stack(Vs, axis=0)  # [T, N, 3]
    T, N, _ = V_all.shape
    V_all_m = matlab.double(V_all.flatten(order="F").tolist(), size=(T, N, 3))
    edges_m = _to_matlab_idx(edge_idx)
    faces_m = _to_matlab_idx(tri_idx)
    bounds_m = matlab.double(list(parsed_bounds))
    view_m = _to_matlab_view(view)

    eng.plot_skeleton(
        "animation", V_all_m, edges_m, faces_m, bounds_m, float(fps), filename,
        "visible", bool(visible),
        "show_floor", bool(show_floor),
        "show_poles", bool(show_poles),
        "show_walls", bool(show_walls),
        "show_axis", bool(show_axis),
        "view_az_el", view_m,
        nargout=0,
    )
    print(f"视频已保存: {filename}")



def compute_rdm(df: pd.DataFrame) -> np.ndarray:
    """
    df: index是17个关节名(如Nose, Head, Neck...)，列是坐标(PC0, PC1, PC2...任意维度)。
    返回: RDM的condensed vector (长度 = 17*16/2 = 136)
    """
    if len(df) != 17:
        raise ValueError(f"期望17个关节，但拿到了{len(df)}个")
    return pdist(df.to_numpy(dtype=float), metric='euclidean')


def compute_rsa_similarity(df1: pd.DataFrame, df2: pd.DataFrame, method='spearman') -> dict:
    """
    计算两个skeleton (df1, df2) 的RSA相似度。
    df1, df2的index(关节名)顺序必须一致，列数可以不同(维度不需要一致)。
    """
    # 保证两边关节顺序对齐（按df1的顺序重排df2）
    df2 = df2.loc[df1.index]

    rdm1 = compute_rdm(df1)
    rdm2 = compute_rdm(df2)

    corr_func = spearmanr if method == 'spearman' else pearsonr
    corr, pval = corr_func(rdm1, rdm2)

    return {'corr': corr, 'pval': pval, 'rdm1': rdm1, 'rdm2': rdm2}


def get_triangular_faces(edges):
    """
    从skeleton edges推导出所有三角形面(3个joint两两相连)。
    这些三角形被视为身体上的"面"，用来检测其他edge是否穿过它们。

    edges: [[a, b], [b, c], ...] 形式的边表
    返回: list of (joint_a, joint_b, joint_c) 三元组
    """
    G = nx.Graph()
    G.add_edges_from(edges)

    faces = []
    nodes = list(G.nodes)
    for a, b, c in combinations(nodes, 3):
        if G.has_edge(a, b) and G.has_edge(b, c) and G.has_edge(a, c):
            faces.append((a, b, c))
    return faces


def segment_triangle_intersect(p0, p1, v0, v1, v2, epsilon=1e-8):
    """
    Möller–Trumbore算法：判断线段 p0->p1 是否穿过三角形 v0,v1,v2。
    返回 (是否相交, 交点坐标 or None)
    """
    d = p1 - p0
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(d, edge2)
    a = np.dot(edge1, h)
    if abs(a) < epsilon:
        return False, None  # 与三角形平行

    f = 1.0 / a
    s = p0 - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False, None

    q = np.cross(s, edge1)
    v = f * np.dot(d, q)
    if v < 0.0 or u + v > 1.0:
        return False, None

    t = f * np.dot(edge2, q)
    # t必须严格落在(0,1)之间，排除端点重合/共享顶点的情况
    if epsilon < t < 1 - epsilon:
        point = p0 + t * d
        return True, point
    return False, None


def compute_crossings(df, edges=None, epsilon=1e-8, return_details=False):
    """
    计算skeleton中每条edge穿过了多少个(由skeleton自身推导出的)三角形面，
    以及具体穿过的三角形和交点坐标。

    df: index是joint名(Nose, Head...)，列是坐标(要求3维)
    edges: 边表，默认调用import_skeleton_config()获取

    返回:
      summary: (int) 穿过次数
      details: {edge(tuple): [{'triangle': (三点), 'point': 交点坐标}, ...]}
    """
    if edges is None:
        edges = import_skeleton_config()

    faces = get_triangular_faces(edges)

    summary = {}
    details = {}
    joints = df.index.to_list()

    for a_name, b_name in edges:
        # if a_name in df.index and b_name in df.index:
        if not {a_name, b_name} <= set(joints):
            continue
        
        p0 = df.loc[a_name].to_numpy(dtype=float)
        p1 = df.loc[b_name].to_numpy(dtype=float)
        edge_key = (a_name, b_name)
        details[edge_key] = []

        for face in faces:
            # edge和face共享顶点时跳过（本来就相连，不算“穿过”）
            if a_name in face or b_name in face:
                continue
            if not set(face) <= set(joints):
                
                continue

            v0 = df.loc[face[0]].to_numpy(dtype=float)
            v1 = df.loc[face[1]].to_numpy(dtype=float)
            v2 = df.loc[face[2]].to_numpy(dtype=float)

            hit, point = segment_triangle_intersect(p0, p1, v0, v1, v2, epsilon)
            if hit:
                details[edge_key].append({'triangle': face, 'point': point})

        summary[edge_key] = len(details[edge_key])

    if return_details:
        return sum(summary.values()), details
    else:
        return sum(summary.values())

def compute_skeleton_prop(df_pc, rdm_r=True, df_pc_vel=None, **kwargs):
    """
    计算骨架属性指标（交叉次数、平均距离、RDM 相关系数）。

    Parameters
    ----------
    df_pc : pandas.DataFrame
        骨架主成分坐标，index 为关节名，列为 PC 维度。
    rdm_r : bool
        若 True，计算与速度 PC 的 RDM 相关系数。
    df_pc_vel : pandas.DataFrame, optional
        速度 PC 数据；None 则自动调用 import_pc 加载。
    **kwargs
        转发给 import_pc 的额外参数。

    Returns
    -------
    pandas.Series
        包含 n_cross、d、rdm_r（可选）等指标的序列。
    """
    prop = {}
    n_cross = compute_crossings(df_pc)
    prop['n_cross'] = n_cross
    prop['d'] = np.mean(compute_rdm(df_pc))

    if rdm_r:
        if df_pc_vel is None:
            df_pc_vel = import_pc(**kwargs)
        res = compute_rsa_similarity(df_pc, df_pc_vel)
        prop['rdm_r'] = res['corr']

    return pd.Series(prop)