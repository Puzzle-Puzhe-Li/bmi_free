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
    """
    逐行归一化 (N,3) 向量。

    Parameters
    ----------
    v : numpy.ndarray, shape (N, 3)
        输入向量数组。

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        归一化后的向量；零向量行返回 NaN。
    """
    n = np.linalg.norm(v, axis=1, keepdims=True)
    with np.errstate(invalid='ignore'):
        return np.where(n > 1e-10, v / n, np.nan)


def make_frame(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """
    构建右手局部坐标系。

    Parameters
    ----------
    primary : numpy.ndarray, shape (N, 3)
        骨骼主轴向量。
    secondary : numpy.ndarray, shape (N, 3)
        横臂向量（无需与 primary 垂直）。

    Returns
    -------
    numpy.ndarray, shape (N, 3, 3)
        列向量为 [e1, e2, e3] 的旋转矩阵。
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

    Parameters
    ----------
    R_prox : numpy.ndarray, shape (N, 3, 3)
        近端旋转矩阵。
    R_dist : numpy.ndarray, shape (N, 3, 3)
        远端旋转矩阵。

    Returns
    -------
    numpy.ndarray, shape (N,)
        相对旋转角（度）。
    """
    R_rel = R_dist @ R_prox.transpose(0, 2, 1)
    cos_val = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))


def bend_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    三点屈曲角（度）：B 为关节，A 近端，C 远端。

    0° = 完全伸展，值越大越弯曲。

    Parameters
    ----------
    A : numpy.ndarray, shape (N, 3)
        近端点坐标。
    B : numpy.ndarray, shape (N, 3)
        关节点坐标。
    C : numpy.ndarray, shape (N, 3)
        远端点坐标。

    Returns
    -------
    numpy.ndarray, shape (N,)
        屈曲角（度）。
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

    Parameters
    ----------
    df_pos : pandas.DataFrame
        位置数据，双层列索引 (bodypart, dim)，dim 为 'x'/'y'/'z'。

    Returns
    -------
    pandas.DataFrame
        关节角度数据，列为各关节名，值为角度（度）。

    Notes
    -----
    局部坐标系定义：

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

    Parameters
    ----------
    df_angle : pandas.DataFrame
        关节角度数据，由 compute_joint_angles 返回。
    df_pos : pandas.DataFrame
        位置数据，用于重建局部坐标系以计算旋转矩阵关节的帧间增量角。

    Returns
    -------
    pandas.DataFrame
        角速度数据，shape 与 df_angle 相同，首行置零。

    Notes
    -----
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

    依赖：mne >= 1.0，edfio（pip install mne edfio）。

    Parameters
    ----------
    df_angle : pandas.DataFrame
        compute_joint_angles() 的返回值，
        index 为时间（秒），列为各关节角（度）。
    output_path : str
        输出 .edf 路径。
    overwrite : bool
        是否覆盖已有文件，默认 True。

    Returns
    -------
    mne.io.RawArray
        同时已写入磁盘的 RawArray 对象。

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


"""
segments.py
-----------
关键点位置 -> 段的局部角速度（大小，deg/s）。

段：Pelvis, Thorax, Head,
    UpperArm(L/R), LowerArm(L/R), UpperLeg(L/R), LowerLeg(L/R)

每个段输出相对其父段的旋转速度大小，Pelvis 相对世界系。
索引 t 处的值是 t-1 到 t 之间的旋转。

    from bmi_free.segments import save_session, import_segment_angvel
    save_session('A0')
    save_session('A0', smooth_win=3)
    df = import_segment_angvel('A0', postfix='_smoothed')
"""

import numpy as np
import pandas as pd

from bmi_free.io import import_pos, path, _sess_str
from bmi_free.features import select_coors


POINT_ALIAS = {'Thigh(L)': 'Hip(L)', 'Thigh(R)': 'Hip(R)'}

SIDES = ('L', 'R')

TRUNK = {
    'Pelvis': dict(points=('Sacrum', 'Back', 'Hip(L)', 'Hip(R)'),
                   axis=('Sacrum', 'Back'), lateral=('Hip(L)', 'Hip(R)')),
    'Thorax': dict(points=('Back', 'Neck', 'Shoulder(L)', 'Shoulder(R)'),
                   axis=('Back', 'Neck'), lateral=('Shoulder(L)', 'Shoulder(R)')),
}

HEAD_POINTS = ('Neck', 'Head', 'Nose')

LIMBS = (
    dict(limb='Forelimb', prox='Shoulder', mid='Elbow', dist='Wrist',
         upper='UpperArm', lower='LowerArm', trunk='Thorax'),
    dict(limb='Hindlimb', prox='Hip', mid='Knee', dist='Ankle',
         upper='UpperLeg', lower='LowerLeg', trunk='Pelvis'),
)

SEGMENTS = ['Pelvis', 'Thorax', 'Head'] + [
    f"{lim[k]}({s})" for lim in LIMBS for k in ('upper', 'lower') for s in SIDES
]


def _build_tables():
    parent = {'Pelvis': None, 'Thorax': 'Pelvis', 'Head': 'Thorax'}
    depends = {
        'Pelvis': list(TRUNK['Pelvis']['points']),
        'Thorax': list(TRUNK['Pelvis']['points']) + list(TRUNK['Thorax']['points']),
        'Head': list(HEAD_POINTS) + list(TRUNK['Thorax']['points']),
    }
    group = {'Pelvis': 'Trunk', 'Thorax': 'Trunk', 'Head': 'Head'}
    for lim in LIMBS:
        trunk_pts = list(TRUNK[lim['trunk']]['points'])
        for s in SIDES:
            p, m, d = (f"{lim[k]}({s})" for k in ('prox', 'mid', 'dist'))
            up, lo = f"{lim['upper']}({s})", f"{lim['lower']}({s})"
            parent[up], parent[lo] = lim['trunk'], up
            depends[up] = [p, m] + trunk_pts
            depends[lo] = [p, m, d] + trunk_pts
            group[up] = group[lo] = lim['limb']
    depends = {k: list(dict.fromkeys(v)) for k, v in depends.items()}
    return parent, depends, group


PARENT, DEPENDS, GROUP = _build_tables()


def _T(R):
    return np.swapaxes(R, -1, -2)


def rot_angle_between(Ra, Rb):
    R = Rb @ _T(Ra)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2],
                  R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    s = np.linalg.norm(v, axis=-1) / 2
    c = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    return np.arctan2(s, c)


def rot_speed(R, dt):
    w = np.full(len(R), np.nan)
    w[1:] = np.degrees(rot_angle_between(R[:-1], R[1:])) / dt
    return w


def vec_speed(U, dt):
    w = np.full(len(U), np.nan)
    a, b = U[:-1], U[1:]
    w[1:] = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1),
                                  np.einsum('ni,ni->n', a, b))) / dt
    return w


def to_local(R, u):
    return np.einsum('nji,nj->ni', R, u)


def kabsch_frames(P, names, primary, secondary, return_resid=False):
    """
    多点刚体拟合，返回 (T, 3, 3) 旋转矩阵（参考系 -> 世界系），无效帧为 NaN。
    参考形状由粗框架（primary/secondary）下各点局部坐标的中位数给出。
    return_resid=True 时同时返回 (T, 点数) 的逐点拟合残差。
    """
    X = np.stack([P[n] for n in names], axis=1)
    Xc = X - X.mean(axis=1, keepdims=True)
    R0 = make_frame(P[primary[1]] - P[primary[0]], P[secondary[1]] - P[secondary[0]])
    valid = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(R0).all(axis=(1, 2))

    local = np.einsum('nji,nkj->nki', R0[valid], Xc[valid])
    ref = np.median(local, axis=0)
    ref = ref - ref.mean(axis=0)

    H = np.einsum('ki,nkj->nij', ref, Xc[valid])
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.tile(np.eye(3), (len(d), 1, 1))
    D[:, 2, 2] = d
    Q = U @ D @ Vt

    R = np.full((len(X), 3, 3), np.nan)
    R[valid] = _T(Q)
    if not return_resid:
        return R
    E = np.full(X.shape[:2], np.nan)
    E[valid] = np.linalg.norm(Xc[valid] - ref @ Q, axis=2)
    return R, E


def trunk_frame(P, name, return_resid=False):
    spec = TRUNK[name]
    return kabsch_frames(P, list(spec['points']), spec['axis'], spec['lateral'], return_resid)


def segment_speeds(P, fps, times=None):
    """返回 (local, world)，两个 DataFrame，列为 SEGMENTS。"""
    dt = 1 / fps
    R_pel = trunk_frame(P, 'Pelvis')
    R_tho = trunk_frame(P, 'Thorax')
    R_skull = make_frame(P['Head'] - P['Neck'], P['Nose'] - P['Head'])
    R_trunk = {'Pelvis': R_pel, 'Thorax': R_tho}

    local, world = {}, {}
    local['Pelvis'] = world['Pelvis'] = rot_speed(R_pel, dt)
    local['Thorax'] = rot_speed(_T(R_pel) @ R_tho, dt)
    world['Thorax'] = rot_speed(R_tho, dt)
    local['Head'] = rot_speed(_T(R_tho) @ R_skull, dt)
    world['Head'] = rot_speed(R_skull, dt)

    for lim in LIMBS:
        R_par = R_trunk[lim['trunk']]
        lat = R_par[:, :, 1]
        for s in SIDES:
            p, m, d = (f"{lim[k]}({s})" for k in ('prox', 'mid', 'dist'))
            up, lo = f"{lim['upper']}({s})", f"{lim['lower']}({s})"
            u = normalize(P[m] - P[p])
            v = normalize(P[d] - P[m])

            local[up] = vec_speed(to_local(R_par, u), dt)
            world[up] = vec_speed(u, dt)

            R_seg = make_frame(u, lat)
            local[lo] = vec_speed(to_local(R_seg, v), dt)
            world[lo] = vec_speed(v, dt)

    local = pd.DataFrame(local)[SEGMENTS]
    world = pd.DataFrame(world)[SEGMENTS]
    if times is not None:
        local.index = world.index = pd.Index(times, name='t')
    return local, world


def valid_mask(P, times=None):
    """每个段在 t 处是否可算：DEPENDS 里的所有点在 t-1 和 t 都有效。"""
    finite = {n: np.isfinite(x).all(axis=1) for n, x in P.items()}
    ok = {}
    for seg in SEGMENTS:
        f = np.logical_and.reduce([finite[n] for n in DEPENDS[seg]])
        v = np.zeros(len(f), bool)
        v[1:] = f[1:] & f[:-1]
        ok[seg] = v
    df = pd.DataFrame(ok)[SEGMENTS]
    if times is not None:
        df.index = pd.Index(times, name='t')
    return df


def load_points(session, smooth_win=None):
    """
    读取 import_pos，按 bad_miss 把缺失帧置 NaN，Thigh 点更名为 Hip。
    smooth_win 给定时，在置 NaN 之后做居中滑动平均；窗口内有缺失的帧置 NaN。
    """
    df = import_pos(session)
    times = df.index.to_numpy(dtype=float)
    P = {}
    for j in df.columns.get_level_values(0).unique():
        x = df[j][['x', 'y', 'z']].to_numpy(dtype=float)
        ok = np.zeros(len(times), bool)
        ok[select_coors(session, j, times)[0]] = True
        x[~ok] = np.nan
        P[POINT_ALIAS.get(j, j)] = x
    if smooth_win:
        P = {j: pd.DataFrame(x).rolling(smooth_win, center=True, min_periods=smooth_win).mean().to_numpy()
             for j, x in P.items()}
    return P, times


def compute_session(session, smooth_win=None):
    P, times = load_points(session, smooth_win)
    fps = 1 / np.median(np.diff(times))
    local, world = segment_speeds(P, fps, times)
    return P, local, world


def segment_angvel_file(session, kind='local', postfix=''):
    name = 'segment_angvel' if kind == 'local' else f'segment_angvel_{kind}'
    return path['coor'] / f'{name}_{_sess_str(session, None)}{postfix}.csv'


def save_session(session, smooth_win=None, postfix=None):
    """写出 local 与 world 两个文件，返回两个路径。"""
    if postfix is None:
        postfix = '_smoothed' if smooth_win else ''
    _, local, world = compute_session(session, smooth_win)
    files = {}
    for kind, df in (('local', local), ('world', world)):
        files[kind] = segment_angvel_file(session, kind, postfix)
        df.to_csv(files[kind])
    return files


def import_segment_angvel(session, kind='local', postfix=''):
    return pd.read_csv(segment_angvel_file(session, kind, postfix), index_col=0)