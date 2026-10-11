"""
kinematics.py
=============
将猴子3D关节追踪坐标（17个关键点）转换为10个主要关节的运动幅度角。

输入
----
df_pos : pd.DataFrame
    由 bmi_free.io.import_joints 返回的位置数据。
    index 为时间，每列为一个关键点名称，值为 (N,3) ndarray。
    关键点名称：Nose, Head, Neck,
               Shoulder(L/R), Elbow(L/R), Wrist(L/R),
               Back, Sacrum,
               Thigh(L/R), Knee(L/R), Ankle(L/R)

输出列（df_angle）
------------------
  Neck          颈关节旋转幅度   [0°,180°]  含低头/仰头/摇头
  Trunk         躯干旋转幅度     [0°,180°]  含前屈/侧弯/扭转
  Shoulder(L/R) 肩关节旋转幅度   [0°,180°]
  Elbow(L/R)    肘屈曲角         [0°,180°]  0=完全伸展
  Thigh(L/R)    髋关节旋转幅度   [0°,180°]
  Knee(L/R)     膝屈曲角         [0°,180°]  0=完全伸展

算法
----
颈/躯干/肩/髋 — 旋转矩阵法
    在关节两侧各建局部坐标系（骨骼轴 + 横臂），
    计算相对旋转矩阵并提取欧拉幅度角：
        θ = arccos((trace(R_rel) − 1) / 2)
    同时包含弯折与轴向旋转，对任意体位均成立，不依赖全局参考系。

肘/膝 — 三点夹角法（缺远端横臂，仅测弯折）
    屈曲角 = 180° − arccos(v₁·v₂)

命名说明
--------
Sacrum 和 Back 都是输入关键点，始终保留在 df_pos 中。
输出列 'Trunk' 是由 Sacrum、Back 两点共同计算出的躯干旋转角，
属于不同层次，不会与任何关键点名冲突。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from bmi_free.io import import_joints


# ─────────────────────────────────────────────────────────────────────────────
# 数学工具
# ─────────────────────────────────────────────────────────────────────────────
AXES = ('x', 'y', 'z')

def cal_vector(
    out: dict,
    bp: str,
    prefix: str,
    data: np.ndarray,
    *,
    write_abs: bool = False,
    write_norm: bool = False,
):
    """
    将 (T, 3) 的向量数据写入输出字典。

    Parameters
    ----------
    out : dict
        输出字典，键为 (bodypart, quantity) 元组。
    bp : str
        身体部位名称。
    prefix : str
        量名称前缀（如 'pos'、'vel'、'acc'）。
    data : numpy.ndarray, shape (T, 3)
        向量时间序列数据。
    write_abs : bool
        若 True，同时写入各分量的绝对值。
    write_norm : bool
        若 True，同时写入向量的范数。
    """

    for i, ax in enumerate(AXES):
        out[(bp, f'{prefix}_{ax}')] = data[:, i]

    if write_abs:
        for i, ax in enumerate(AXES):
            out[(bp, f'{prefix}_{ax}_abs')] = np.abs(data[:, i])

    if write_norm:
        out[(bp, f'{prefix}_norm')] = np.linalg.norm(data, axis=1)

def diff(arr: np.ndarray, dt: float) -> np.ndarray:
    """
    沿 axis=0 的前向差分（第一行补零）。

    Parameters
    ----------
    arr : numpy.ndarray
        输入数组，shape (T, ...)。
    dt : float
        时间步长。

    Returns
    -------
    numpy.ndarray
        差分结果，shape 与 arr 相同，第一行为零。
    """
    return np.vstack([
        np.zeros(arr.shape[1]),
        np.diff(arr, axis=0) / dt
    ])


def cal_kinematics(df_pos: pd.DataFrame, fps: int = 30) -> pd.DataFrame:
    """
    计算各身体部位的位置、速度和加速度。

    Parameters
    ----------
    df_pos : pandas.DataFrame
        位置数据，双层列索引 (bodypart, dim)，dim 为 'x'/'y'/'z'。
    fps : int
        采样帧率，默认 30。

    Returns
    -------
    pandas.DataFrame
        运动学数据，双层列索引 (bodypart, quantity)，包含 pos/vel/acc
        各分量及其绝对值和范数。
    """
    dt = 1 / fps
    bodyparts = df_pos.columns.get_level_values(0).unique()
    out = {}

    for bp in bodyparts:
        # position
        pos = df_pos[bp][list(AXES)].values
        cal_vector(out, bp, 'pos', pos)

        # velocity
        vel = diff(pos, dt)
        cal_vector(out, bp, 'vel', vel, write_abs=True, write_norm=True)

        # acceleration
        acc = diff(vel, dt)
        cal_vector(out, bp, 'acc', acc, write_abs=True, write_norm=True)

    df_out = pd.DataFrame(out, index=df_pos.index)
    df_out.columns = pd.MultiIndex.from_tuples(
        df_out.columns,
        names=['bodypart', 'quantity']
    )

    return df_out

def normalize(v: np.ndarray) -> np.ndarray:
    """逐行归一化 (N,3) → (N,3)，零向量行返回 nan。"""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    with np.errstate(invalid='ignore'):
        return np.where(n > 1e-10, v / n, np.nan)


def make_frame(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """
    构建右手局部坐标系。

    Parameters
    ----------
    primary   : (N,3) 骨骼主轴向量
    secondary : (N,3) 横臂向量（无需与 primary 垂直）

    Returns
    -------
    R : (N,3,3)  列向量为 [e1, e2, e3] 的旋转矩阵
    """
    e1 = normalize(primary)
    proj = np.einsum('ni,ni->n', secondary, e1)[:, None] * e1
    e2 = normalize(secondary - proj)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=2)


def rot_angle(R_prox: np.ndarray, R_dist: np.ndarray) -> np.ndarray:
    """
    两坐标系间的相对旋转幅度角（度）。

    R_rel = R_dist @ R_prox^T
    θ = arccos((trace(R_rel) − 1) / 2)   ∈ [0°, 180°]
    """
    R_rel = R_dist @ R_prox.transpose(0, 2, 1)
    cos_val = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))


def bend_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    三点屈曲角（度）：B 为关节，A 近端，C 远端。
    0° = 完全伸展，值越大越弯曲。
    """
    v1 = normalize(A - B)
    v2 = normalize(C - B)
    cos_val = np.clip(np.einsum('ni,ni->n', v1, v2), -1.0, 1.0)
    return 180.0 - np.degrees(np.arccos(cos_val))


# ─────────────────────────────────────────────────────────────────────────────
# 关节角度计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_joint_angles(df_pos: pd.DataFrame) -> pd.DataFrame:
    """
    输入位置 DataFrame，返回关节角度 DataFrame。

    局部坐标系定义
    ──────────────
    颈 (Neck)
        近端：骨骼轴 = Neck−Back，        横臂 = Shoulder(R)−Shoulder(L)
        远端：骨骼轴 = Head−Neck，         横臂 = Nose−Head
    躯干 (Trunk)
        近端：骨骼轴 = Back−Sacrum，       横臂 = Thigh(R)−Thigh(L)
        远端：骨骼轴 = Neck−Back，         横臂 = Shoulder(R)−Shoulder(L)
    肩 Shoulder(L/R)
        近端：同躯干远端框架
        远端：骨骼轴 = Elbow−Shoulder，    横臂 = Wrist−Elbow
    肘 Elbow(L/R)   三点角：Shoulder → Elbow → Wrist
    髋 Thigh(L/R)
        近端：同躯干近端框架
        远端：骨骼轴 = Knee−Thigh，        横臂 = Ankle−Knee
    膝 Knee(L/R)    三点角：Thigh → Knee → Ankle
    """
    p = df_pos  # 简写

    # 复用公共框架
    R_trunk_lower = make_frame(p['Back']   - p['Sacrum'],
                               p['Thigh(R)'] - p['Thigh(L)'])
    R_trunk_upper = make_frame(p['Neck']   - p['Back'],
                               p['Shoulder(R)'] - p['Shoulder(L)'])

    out = {}

    # 颈
    R_neck_prox = make_frame(p['Neck'] - p['Back'],
                              p['Shoulder(R)'] - p['Shoulder(L)'])
    R_neck_dist = make_frame(p['Head'] - p['Neck'],
                              p['Nose'] - p['Head'])
    out['Neck'] = rot_angle(R_neck_prox, R_neck_dist)

    # 肩
    for side in ['L', 'R']:
        R_arm = make_frame(p[f'Elbow({side})']   - p[f'Shoulder({side})'],
                           p[f'Wrist({side})']    - p[f'Elbow({side})'])
        out[f'Shoulder({side})'] = rot_angle(R_trunk_upper, R_arm)

    # 肘
    for side in ['L', 'R']:
        out[f'Elbow({side})'] = bend_angle(
            p[f'Shoulder({side})'], p[f'Elbow({side})'], p[f'Wrist({side})'])
        
    # 背
    out['Back'] = rot_angle(R_trunk_lower, R_trunk_upper)

    # 骶
    hip_mid = (p['Thigh(L)'] + p['Thigh(R)']) / 2
    R_pelvis     = make_frame(p['Sacrum'] - hip_mid,
                            p['Thigh(R)'] - p['Thigh(L)'])
    R_lower_spine = make_frame(p['Back'] - p['Sacrum'],
                                p['Thigh(R)'] - p['Thigh(L)'])
    out['Sacrum'] = rot_angle(R_pelvis, R_lower_spine)

    # 髋
    for side in ['L', 'R']:
        R_leg = make_frame(p[f'Knee({side})']  - p[f'Thigh({side})'],
                           p[f'Ankle({side})'] - p[f'Knee({side})'])
        out[f'Thigh({side})'] = rot_angle(R_trunk_lower, R_leg)

    # 膝
    for side in ['L', 'R']:
        out[f'Knee({side})'] = bend_angle(
            p[f'Thigh({side})'], p[f'Knee({side})'], p[f'Ankle({side})'])

    return pd.DataFrame(out, index=df_pos.index)

def compute_angle_vel(df_angle: pd.DataFrame,
                          df_pos: pd.DataFrame) -> pd.DataFrame:
    """
    计算各关节的角速度大小（deg/s），恒 ≥ 0。

    对颈/躯干/肩/髋（旋转矩阵关节）：
        用帧间增量旋转角 Δθ/Δt，天然非负，无需取绝对值。
    对肘/膝（三点弯折角关节）：
        用 |dθ/dt|，同样非负。
    """
    t  = df_angle.index.to_numpy(dtype=float)
    dt = np.diff(t)[0]                    # (N-1,)
    p  = df_pos

    df_vel = df_angle.copy()
    df_vel.iloc[0] = 0

    # speed = {}

    # ── 旋转矩阵关节：帧间增量旋转角 ──────────────────────────────
    rot_joints = {
        'Neck': (
            make_frame(p['Neck'] - p['Back'],
                       p['Shoulder(R)'] - p['Shoulder(L)']),
            make_frame(p['Head'] - p['Neck'],
                       p['Nose'] - p['Head']),
        ),
        'Back': (
            make_frame(p['Back']   - p['Sacrum'],
                       p['Thigh(R)'] - p['Thigh(L)']),
            make_frame(p['Neck']   - p['Back'],
                       p['Shoulder(R)'] - p['Shoulder(L)']),
        ),
        'Sacrum': (
            make_frame(p['Sacrum'] - (p['Thigh(L)']+p['Thigh(R)'])/2,
                       p['Thigh(R)'] - p['Thigh(L)']),
            make_frame(p['Back']   - p['Sacrum'],
                       p['Thigh(R)'] - p['Thigh(L)']),
        ),
    }
    for side in ['L', 'R']:
        R_trunk_upper = make_frame(p['Neck'] - p['Back'],
                                   p['Shoulder(R)'] - p['Shoulder(L)'])
        R_trunk_lower = make_frame(p['Back'] - p['Sacrum'],
                                   p['Thigh(R)'] - p['Thigh(L)'])
        rot_joints[f'Shoulder({side})'] = (
            R_trunk_upper,
            make_frame(p[f'Elbow({side})']   - p[f'Shoulder({side})'],
                       p[f'Wrist({side})']    - p[f'Elbow({side})']),
        )
        rot_joints[f'Thigh({side})'] = (
            R_trunk_lower,
            make_frame(p[f'Knee({side})']  - p[f'Thigh({side})'],
                       p[f'Ankle({side})'] - p[f'Knee({side})']),
        )

    for name, (R_prox, R_dist) in rot_joints.items():
        R_rel = R_dist @ R_prox.transpose(0, 2, 1)          # (N,3,3)
        # 帧间增量旋转
        dR       = R_rel[1:] @ R_rel[:-1].transpose(0, 2, 1)
        cos_val  = (np.trace(dR, axis1=1, axis2=2) - 1) / 2
        delta_th = np.degrees(np.arccos(np.clip(cos_val, -1, 1)))
        # speed[name] = delta_th / dt                          # deg/s, ≥ 0
        df_vel.loc[df_vel.index[1:], name] = delta_th / dt                       # deg/s, ≥ 0

    # ── 三点弯折关节：|dθ/dt| ──────────────────────────────────────
    bend_joints = ['Elbow(L)', 'Elbow(R)', 'Knee(L)', 'Knee(R)']
    df_vel.loc[df_vel.index[1:], bend_joints] = np.abs(np.diff(df_angle[bend_joints], axis=0)) / dt           # deg/s, ≥ 0
    # bend_joints = {
    #     'Elbow(L)': df_angle['Elbow(L)'].to_numpy(),
    #     'Elbow(R)': df_angle['Elbow(R)'].to_numpy(),
    #     'Knee(L)':  df_angle['Knee(L)'].to_numpy(),
    #     'Knee(R)':  df_angle['Knee(R)'].to_numpy(),
    # }
    # for name, theta in bend_joints.items():
    #     df_vel[name] = np.abs(np.diff(theta)) / dt           # deg/s, ≥ 0

    # df_angle_vel = df_angle.copy()
    # df_angle_vel = pd.DataFrame(speed, index=df_angle.index[1:]).sort_index(columns=True)
    # df_angle_vel.iloc[:1] = 0
    
    return df_vel



def quan_to_edf(df_angle: pd.DataFrame, output_path: str,
                  overwrite: bool = True):
    """
    将关节角度 DataFrame 写为 EDF 文件。
 
    依赖：mne >= 1.0，edfio（pip install mne edfio）
 
    Parameters
    ----------
    df_angle    : compute_joint_angles() 的返回值
                  index 为时间（秒），列为各关节角（度）
    output_path : 输出 .edf 路径
    overwrite   : 是否覆盖已有文件，默认 True
 
    Returns
    -------
    raw : mne.io.RawArray（同时已写入磁盘）
 
    Notes
    -----
    EDF 格式的两项硬性限制，函数会自动处理：
    1. 采样率必须为整数 → 对中位采样间隔取整。
    2. 总帧数必须能被采样率整除（即总时长为整数秒）
       → 末尾多余帧会被截掉（通常 < 1 秒）。
    MNE 对 misc 通道固定写 "uV" 作为 physical_dimension，
    函数在写完后用 edfio 将其修正为 "deg"。
    """
    import mne
 
    # ── 采样率 ────────────────────────────────────────────────────
    t     = df_angle.index.to_numpy(dtype=float)
    diffs = np.diff(t)
    sfreq_raw = 1.0 / np.median(diffs)
 
    jitter = np.std(diffs) / np.median(diffs)
    if jitter > 0.01:
        print(f"[警告] 时间戳不均匀（抖动 {jitter*100:.1f}%），"
              f"按中位采样率 {sfreq_raw:.2f} Hz 存储。")
 
    sfreq = round(sfreq_raw)   # EDF 要求整数采样率
 
    # ── 数据截断到整秒 ─────────────────────────────────────────────
    data = df_angle.to_numpy(dtype=np.float64).T          # (n_ch, n_times)
    n_complete = (data.shape[1] // sfreq) * sfreq
    if n_complete < data.shape[1]:
        print(f"[info] 末尾 {data.shape[1] - n_complete} 帧截断"
              f"（EDF 要求总时长为整数秒）。")
    data = data[:, :n_complete]
 
    # ── 构建 MNE RawArray ──────────────────────────────────────────
    ch_names = list(df_angle.columns)
    info = mne.create_info(
        ch_names = ch_names,
        sfreq    = float(sfreq),
        ch_types = ['misc'] * len(ch_names),
        verbose  = False,
    )

    info.set_meas_date(None)
 
    raw = mne.io.RawArray(data, info, verbose=False)
 
    # ── 导出 EDF ───────────────────────────────────────────────────
    mne.export.export_raw(
        output_path, raw, fmt='edf',
        overwrite=overwrite, verbose=False,
    )
 
    return raw
 

# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

# def process_file(csv_path: str, output_path: str | None = None) -> pd.DataFrame:
#     """
#     读取一个追踪CSV，计算关节角，保存并返回结果 DataFrame。

#     Parameters
#     ----------
#     csv_path    : 输入 CSV 路径
#     output_path : 输出 CSV 路径（默认在原文件同目录，后缀改为 _angles.csv）
#     """
#     if output_path is None:
#         p = Path(csv_path)
#         output_path = str(p.parent / (p.stem + '_angles.csv'))

#     df_pos    = import_joints(csv_path)
#     df_angles = compute_joint_angles(df_pos)
#     df_angles.to_csv(output_path, float_format='%.4f')

#     print(f"[✓] {Path(csv_path).name}  →  {Path(output_path).name}"
#           f"  ({len(df_angles)} 帧)")
#     return df_angles


# if __name__ == '__main__':
#     import sys
#     if len(sys.argv) < 2:
#         print("用法: python kinematics.py <input.csv> [output.csv]")
#         sys.exit(1)
#     process_file(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)