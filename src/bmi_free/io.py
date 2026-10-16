import numpy as np
import pandas as pd
from pathlib import Path
from decouple import config

# 建立文件路径字典，键为文件夹名，值为文件夹路径
path_proj = Path(config("PROJ_DATA_DIR"))
path = {}
for folder in path_proj.iterdir():
    if folder.is_dir():
        path[folder.name] = folder

# path_features = Path(r'Y:\decoder_ana\features')
# path_prep = Path(r'Y:\decoder_ana\ecog')
# path_coor = Path(r'Y:\decoder_ana\coor')
# path_config = Path(r'Y:\decoder_ana\config')

sess_info = {
    '399': '20250717-0',
    # '539': '20250819-0',
    '539': '20250819-1',
}

monkey_alias = {
    '399': 'A',
    '539': 'B',
}

monkey_code = {
    'A': '399',
    'B': '539',
}

sessions = ['A0', 'B1']

def sn2loc(sn):
    """
    将相机序列号(sn)映射到对应的拍摄位置(location)。

    Parameters
    ----------
    sn : str or list of str
        单个 sn 值或 sn 列表/数组。

    Returns
    -------
    str or list of str
        对应的 location 值或 location 列表/数组。
    """
    df = import_camera_info()
    df = df.set_index('sn')
    return df.loc[sn, 'location']

def loc2sn(loc):
    """
    将拍摄位置(location)映射到对应的相机序列号(sn)。

    Parameters
    ----------
    loc : str or list of str
        单个 location 值或 location 列表/数组。

    Returns
    -------
    str or list of str
        对应的 sn 值或 sn 列表/数组。
    """
    df = import_camera_info()
    df = df.set_index('location')
    return df.loc[loc, 'sn']

def check_tag(tag):
    """
    确保标签字符串以下划线开头。

    Parameters
    ----------
    tag : str
        标签字符串。

    Returns
    -------
    str
        若 tag 非空且不以 '_' 开头，则在前面补 '_'；否则原样返回。
    """
    if tag != "" and tag[0] != "_":
        tag = "_" + tag

    return tag


def import_camera_info():
    """
    导入相机信息表。

    Returns
    -------
    pandas.DataFrame
        相机信息表，sn 列为 str 类型。
    """
    df = pd.read_excel(path['config'] / 'camera_info.xlsx', index_col=0, dtype={'sn': str}) 
    return df

def import_fps(session, monkey=None, stage='quan'):
    """
    导入帧率。

    Parameters
    ----------
    session : str
        会话标识符，如 'A0'、'B1'。
    monkey : str, optional
        动物标识符（保留用于签名兼容）。
    stage : {'pos', 'quan', 'update', 'setting'}
        计算帧率的阶段：
        - 'pos' : 从位置数据计算
        - 'quan' : 从量化坐标数据计算
        - 'update' : 从会话信息中读取
        - 'setting' : 使用固定值 30

    Returns
    -------
    float
        帧率 (Hz)。
    """
    if stage == 'pos':
        df_pos = import_pos(session=session)
        fps = (df_pos.shape[0] - 1) / (df_pos.index[-1] - df_pos.index[0])
    elif stage == 'quan':
        df_quan, _, _ = import_cartesian_quan(session=session)
        fps = (df_quan.shape[0] - 1) / (df_quan.index[-1] - df_quan.index[0])       
    elif stage == 'update':
        fps = import_session_info(session)['fps']
    elif stage == 'setting':
        fps = 30
    else:
        assert "stage error"
        
    return fps


def import_aligned_frames(session, monkey=None):
    """
    导入合并后的帧编号表。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符（保留用于签名兼容）。

    Returns
    -------
    pandas.DataFrame
        合并后的帧数据框，列名为 location。
    """
    fname_aligned = path['raw'] / session / f'aligned_frames.csv'
    df_aligned = pd.read_csv(fname_aligned, index_col=0).rename(columns=sn2loc)
    return df_aligned

def import_aligned_timestamps(session, monkey=None):
    """
    导入合并后的时间戳表。

    根据各相机的帧编号-时间戳映射，将帧编号转换为绝对时间戳，
    并将起始时间归零。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符（保留用于签名兼容）。

    Returns
    -------
    pandas.DataFrame
        合并后的时间戳数据框，列名为 location，时间从 0 开始。
    """
    df_frames = import_aligned_frames(session, monkey)
    df_timestamps = df_frames.copy().astype(float)
    for loc in df_frames.columns:
        sn = loc2sn(loc)
        df_tmp = pd.read_csv(path['raw'] / session / f"{sn}.csv")
        df_tmp["frame_number"] = df_tmp["frame_number"] - df_tmp["frame_number"].min()
        mapping = df_tmp.set_index("frame_number")["timestamp"]
        df_timestamps[loc] = df_frames[loc].map(mapping)

    df_timestamps = df_timestamps - df_timestamps.min(axis=None)
    return df_timestamps

def import_session_info(session=None, if_select=True):
    """
    导入会话信息表。

    Parameters
    ----------
    session : str, optional
        若提供，则返回该会话对应的一行；否则返回整个表。
    if_select : bool
        若 True，则只保留 selected == 1 的行。

    Returns
    -------
    pandas.DataFrame or pandas.Series
        会话信息表或单行。
    """
    df = pd.read_excel(path['config'] / 'sess_info.xlsx', index_col=0)

    if if_select:
        df = df.query('selected == 1')

    if session is not None:
        df = df.loc[session]

    return df

def import_montage():
    """
    导入 ECoG 64 通道电极布局 (MNE DigMontage)。

    Returns
    -------
    mne.channels.DigMontage
        电极布局对象。
    """
    from mne.channels import read_dig_fif
    montage = read_dig_fif(path['config'] / r'mne_layout_ecog64_20250606-dig.fif')
    return montage

def get_bdf_start_time(path_bdf):
    """
    获取 BDF 文件的起始时间。

    Parameters
    ----------
    path_bdf : str or Path
        BDF 文件路径。

    Returns
    -------
    datetime.datetime
        BDF 文件的起始时间。
    """
    from pyedflib import EdfReader

    with EdfReader(str(path_bdf)) as f:
        # 获取开始时间（datetime 对象）
        start_time = f.getStartdatetime()  # 返回 datetime.datetime
        start_time = start_time.replace(microsecond=start_time.microsecond * 10) # 似乎解析规则不一样，所以这里要乘以10
    return start_time

def import_ch_names():
    """
    导入 ECoG 通道名称列表（排除包含 '11' 的 GND/REF 通道）。

    Returns
    -------
    list of str
        通道名称列表。
    """
    from mne.channels import read_dig_fif
    montage = read_dig_fif(path['config'] / 'mne_layout_ecog64_20250606-dig.fif')
    ch_names = montage.ch_names
    ch_names = [ch for ch in ch_names if '11' not in ch]

    return ch_names

def import_elec_loc(postfix='_64_20250606'):
    """
    导入电极位置信息表。

    Parameters
    ----------
    postfix : str
        文件名后缀，默认 '_64_20250606'。

    Returns
    -------
    pandas.DataFrame
        电极位置信息表。
    """
    df_elec_loc = pd.read_excel(path['config'] / f'elec_loc{postfix}.xlsx', index_col=0)

    return df_elec_loc

def import_config_skeleton_file():
    """
    导入骨架配置文件 (YAML)。

    Returns
    -------
    dict
        骨架配置字典，包含 'joints' 和 'skeleton' 等键。
    """
    from yaml import safe_load
    path_yaml = path['config'] / "config_skeleton.yaml"
    with open(path_yaml, 'r', encoding='utf-8') as f:
        config = safe_load(f)    
        return config
    
def import_joints(categories=True):
    """
    导入关节名称列表。

    Parameters
    ----------
    categories : bool
        若 True，返回 pd.Categorical（保持定义顺序）；否则返回普通列表。

    Returns
    -------
    pandas.Categorical or list
        关节名称列表。
    """
    config = import_config_skeleton_file()
    joints = config['joints']
    if categories:
        joints = pd.Categorical(joints, categories=joints, ordered=True)
    
    return joints

def import_skeleton_config():
    """
    导入骨架连接关系（边表）。

    Returns
    -------
    list of list
        骨架连接关系，每个元素为 [joint_a, joint_b]。
    """
    config = import_config_skeleton_file()
    edges = config['skeleton']
    return edges

def import_annots(session=None, raw=None):
    """
    导入 MNE Annotations 对象。

    Parameters
    ----------
    session : str, optional
        会话标识符。若提供，则从预处理后的 ECoG 数据中读取 annotations。
    raw : mne.io.Raw, optional
        若提供，则直接从 raw.annotations 获取。

    Returns
    -------
    mne.Annotations
        标注对象。

    Raises
    ------
    ValueError
        若 session 和 raw 均未提供。
    """
    if session is not None:
        raw = import_ecog_prep(session)
        annots = raw.annotations
    elif raw is not None:
        annots = raw.annotations
    elif annots is None:
        raise ValueError('Please specify either session or raw')
    
    return annots

def import_onsets(session=None, raw=None, annots=None):
    """
    导入标注的起始时间字典。

    Parameters
    ----------
    session : str, optional
        会话标识符。
    raw : mne.io.Raw, optional
        原始数据对象。
    annots : mne.Annotations, optional
        直接传入 annotations 对象。

    Returns
    -------
    pandas.Series
        以 description 为索引、onset 为值的序列。

    Raises
    ------
    ValueError
        若 session、raw、annots 均未提供。
    """
    if (session is not None) or (raw is not None):
        annots = import_annots(session=session, raw=raw)

    elif annots is None:
        raise ValueError('Please specify either session or raw')
    
    onsets = annots.to_data_frame(time_format=None).set_index('description')['onset']
    
    return onsets

def import_ecog_prep(session, monkey=None, postfix='_prep', **kwargs):
    """
    导入预处理后的 ECoG 数据 (BDF 格式)。

    自动裁剪到 BAD_ACQ_SKIP 标注之前，并设置电极布局。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符（保留用于签名兼容）。
    postfix : str
        文件名后缀，默认 '_prep'。
    **kwargs
        转发给 mne.io.read_raw_bdf 的额外参数。

    Returns
    -------
    mne.io.Raw
        预处理后的 ECoG 数据，info['description'] 设为 session。
    """
    
    from mne.io import read_raw_bdf

    def _get_tmax():
        tmax = raw.annotations.to_data_frame(time_format=None).set_index('description')['onset']['BAD_ACQ_SKIP'].min()
        return tmax
    
    fname = path['ecog'] /f'ecog_{session}{postfix}.bdf'
    raw = read_raw_bdf(fname, verbose='CRITICAL', **kwargs)
    raw.crop(tmax=_get_tmax(), include_tmax=False)
    raw.set_montage(import_montage(), verbose='CRITICAL')
    raw.info['description'] = session
    return raw

def import_bad(session):
    """
    导入坏段标注表（仅包含 Annotation 含 'bad' 的行）。

    时间已相对于 sample_start 归零。

    Parameters
    ----------
    session : str
        会话标识符。

    Returns
    -------
    pandas.DataFrame
        坏段标注表，包含 Onset、Duration、Annotation 列。
    """
    fname = path['ecog'] / f'ecog_{session}_cropped_filtered_rereferred_+ic_annotations.csv'
    df = pd.read_csv(fname)
    if df.query('Annotation == "sample_start"').empty:
        Warning('No time of sample start')
        df = pd.concat([df, pd.DataFrame([{'Onset': 1, 'Duration': 0, 'Annotation': 'sample_start'}])], axis=0)

    sample_start = df.loc[df['Annotation'] == 'sample_start']['Onset'].values
    df['Onset'] -= sample_start
    df = df[df['Annotation'].str.contains('bad', case=False, na=False, regex=False)]

    return df

def import_miss(session, monkey=None, query=None, pad=0):
    """
    导入缺失标注表 (bad_miss)。

    Parameters
    ----------
    session : str
        会话标识符，如 'A0' 或 'B1'。
    monkey : str, optional
        动物标识符（保留用于签名兼容）。
    query : str, optional
        若提供，则对 DataFrame 执行 df.query(query) 筛选。
    pad : float
        对每段缺失区间向两侧扩展的时间（秒），Onset 减半、Duration 增加。

    Returns
    -------
    pandas.DataFrame
        包含 Onset、Duration、Joint 列的缺失标注表。
    """
    fname = path['coor'] / f'annots_{session}_bad_miss.xlsx'

    df = pd.read_excel(fname)

    df['Onset'] = df['Onset'].astype(float) - pad / 2
    df['Duration'] = df['Duration'].astype(float) + pad
    df['Joint'] = df['Joint'].astype(str)
    if query is not None:
        df = df.query(query)

    return df


def import_miss_annots(session, joint=None, query=None, description='bad_miss', orig_time=None):
    """
    将缺失标注表转换为 MNE Annotations 对象。

    Parameters
    ----------
    session : str
        会话标识符。
    joint : str, optional
        若提供，则只加载该关节的缺失段。
    query : str, optional
        筛选查询表达式。
    description : str
        标注描述前缀，默认 'bad_miss'。
    orig_time : float, optional
        Annotations 的 orig_time。

    Returns
    -------
    mne.Annotations
        缺失标注对象，ch_names 为各关节名。
    """
    import mne

    df = import_miss(session, query=query)
    if joint is not None:
        df = df[df['Joint'] == joint].copy()

    if df.empty:
        return mne.Annotations(onset=[], duration=[], description=[])

    descr = [description] * len(df)
    return mne.Annotations(
        onset=df['Onset'].to_numpy(dtype=float),
        duration=df['Duration'].to_numpy(dtype=float),
        description=np.asarray(descr, dtype=str),
        ch_names=[ [j] for j in df['Joint'].tolist() ],
        orig_time=orig_time
    )

def import_involvement(session, query=None, pad=0, postfix="_sorted"):
    """
    导入肢体参与度标注表。

    Parameters
    ----------
    session : str
        会话标识符。
    query : str, optional
        若提供，则对 DataFrame 执行 df.query(query) 筛选。
    pad : float
        对每段区间向两侧扩展的时间（秒）。
    postfix : str
        文件名后缀，默认 '_sorted'。

    Returns
    -------
    pandas.DataFrame
        参与度标注表，包含 Onset、Duration、Annotation 列。
    """

    fname = path['videos'] / f'involvement_{session}{postfix}.csv'
    df = pd.read_csv(fname)

    if not df.query('Annotation == "sample_start"').empty:
        sample_start = df.loc[df['Annotation'] == 'sample_start']['Onset'].values
        df['Onset'] -= sample_start
    else:
        assert('No time of sample start')    

    df['Onset'] = df['Onset'].astype(float) - pad / 2
    df['Duration'] = df['Duration'].astype(float) + pad
    df['Annotation'] = df['Annotation'].astype(str)
    if query is not None:
        df = df.query(query)
    return df

def import_ecog_features(session, monkey=None, fn_postfix='', num_packages=None):
    """
    导入 ECoG 特征数据（MAT 格式，支持分包读取）。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符。
    fn_postfix : str
        文件名后缀，若不以 '_' 开头会自动补 '_'。
    num_packages : int, optional
        最多读取的数据包数量；None 表示读取全部。

    Returns
    -------
    numpy.ndarray
        合并后的特征数组，dtype 为 float32。
    """
    from scipy.io import loadmat

    sess_str = _sess_str(session, monkey)

    # 检查fn_postfix是否为空字符串，如果不为空且不是以_开头，则添加_
    if fn_postfix and not fn_postfix.startswith('_'):
        fn_postfix = '_' + fn_postfix
    
    # 生成第一个包的文件名
    base_filename = f"ecog_features_{sess_str}"
    if fn_postfix:
        base_filename += fn_postfix
    
    # 读取数据包
    features = []
    i = 0
    while True:
        # 生成当前包的文件名
        if i == 0:
            filename = f"{base_filename}.mat"
        else:
            filename = f"{base_filename}_{i}.mat"
        
        # 拼接文件路径
        filepath = path['features'] / filename

        # 检查文件是否存在
        if not filepath.exists():
            print(f'File {filepath} does not exist.')
            break  # 如果没有文件，则停止读取
        
        # 加载mat文件并提取数据
        mat_data = loadmat(filepath)
        features_pkg = mat_data['features_pkg']
        
        # 将当前包的数据加入到features中
        features.append(features_pkg)
        
        i += 1
        
        # 如果指定了读取数量，则停止
        if num_packages and i >= num_packages:
            break

    # 将所有包合并成一个大数组
    return np.concatenate(features, axis=0).astype('float32')

def import_pos(session, monkey=None, postfix=""):
    """
    导入原始位置数据。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符。
    postfix : str
        文件名后缀。

    Returns
    -------
    pandas.DataFrame
        位置数据，双层列索引 (joint, dim)。
    """
    sess_str = _sess_str(session, monkey)
    pos_file = path['coor'] / f'pos_{sess_str}{postfix}.csv'
    df_pos = pd.read_csv(pos_file, header=[0, 1], index_col=0)

    return df_pos

def import_coor(session, monkey=None, postfix="_standardized"):
    """
    导入标准化后的坐标数据。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符。
    postfix : str
        文件名后缀，默认 '_standardized'。

    Returns
    -------
    pandas.DataFrame
        坐标数据，双层列索引 (joint, dim)。
    """
    sess_str = _sess_str(session, monkey)
    coor_file = path['coor'] / f'coor_{sess_str}{postfix}.csv'
    df_coor = pd.read_csv(coor_file, header=[0, 1], index_col=0)

    return df_coor

def import_cartesian_quan(session, monkey=None, quans=None):
    """
    导入笛卡尔坐标量化数据。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符。
    quans : list of str, optional
        需要保留的量化维度；None 表示保留全部。

    Returns
    -------
    df_quan : pandas.DataFrame
        量化数据，双层列索引 (joint, quan)。
    joints : list of str
        关节名称列表。
    quans : list of str
        量化维度列表。
    """
    sess_str = _sess_str(session, monkey)
    df_quan = pd.read_csv(path['coor'] / f'cartesian_quan_{sess_str}.csv', index_col=0, header=[0, 1])
    joints = df_quan.columns.get_level_values(0).unique().to_list()
    if quans is None:
        quans = df_quan.columns.get_level_values(1).unique().to_list()
    else:
        df_quan = df_quan.xs(quans, axis=1, level=1)
    return df_quan, joints, quans

def import_euler_angle_quan(session, monkey=None, quans=None):
    """
    导入欧拉角量化数据。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符。
    quans : list of str, optional
        需要保留的量化维度；None 表示保留全部。

    Returns
    -------
    df_quan : pandas.DataFrame
        量化数据，双层列索引 (joint, quan)。
    joints : list of str
        关节名称列表。
    quans : list of str
        量化维度列表。
    """
    sess_str = _sess_str(session, monkey)
    df_quan = pd.read_csv(path['coor'] / f'euler_angle_quan_{sess_str}.csv', index_col=0, header=[0, 1])
    joints = df_quan.columns.get_level_values(0).unique().to_list()
    if quans is None:
        quans = df_quan.columns.get_level_values(1).unique().to_list()
    else:
        df_quan = df_quan.xs(quans, axis=1, level=1)
    return df_quan, joints, quans

def import_cartesian_vel_norm(session, monkey=None, postfix="", return_joints=False):
    """
    导入笛卡尔速度范数数据。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符。
    postfix : str
        文件名后缀。
    return_joints : bool
        若 True，同时返回关节名称列表。

    Returns
    -------
    pandas.DataFrame or tuple
        速度范数数据；若 return_joints=True，返回 (df, joints)。
    """
    sess_str = _sess_str(session, monkey)
    df_vel_norm = pd.read_csv(path['coor'] / f'cartesian_vel_norm_{sess_str}{postfix}.csv', index_col=0)
    # joints = df_vel_norm.columns.get_level_values(0).unique().to_list()
    if return_joints:
        joints = import_joints()
        return df_vel_norm, joints
    else:
        return df_vel_norm

def import_angle_vel(session, monkey=None, prefix="angle_vel", postfix="_smoothed", return_joints=False):
    """
    导入角度速度数据。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符。
    prefix : str
        文件名前缀，默认 'angle_vel'。
    postfix : str
        文件名后缀，默认 '_smoothed'。
    return_joints : bool
        若 True，同时返回关节名称列表。

    Returns
    -------
    pandas.DataFrame or tuple
        角度速度数据；若 return_joints=True，返回 (df, joints)。
    """
    sess_str = _sess_str(session, monkey)
    df_vel_norm = pd.read_csv(path['coor'] / f'{prefix}_{sess_str}{postfix}.csv', index_col=0)
    # joints = df_vel_norm.columns.get_level_values(0).unique().to_list()
    if return_joints:
        joints = import_joints()
        return df_vel_norm, joints
    else:
        return df_vel_norm

def import_kin_quan(session, monkey=None, quan="cartesian_vel_norm", postfix="_smoothed", return_joints=False):
    """
    导入运动学量化数据的通用接口。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符。
    quan : str
        量化类型名称，用作文件名前缀，默认 'cartesian_vel_norm'。
    postfix : str
        文件名后缀，默认 '_smoothed'。
    return_joints : bool
        若 True，同时返回关节名称列表。

    Returns
    -------
    pandas.DataFrame or tuple
        运动学量化数据；若 return_joints=True，返回 (df, joints)。
    """
    sess_str = _sess_str(session, monkey)
    df_vel_norm = pd.read_csv(path['coor'] / f'{quan}_{sess_str}{postfix}.csv', index_col=0)
    # joints = df_vel_norm.columns.get_level_values(0).unique().to_list()
    if return_joints:
        joints = import_joints()
        return df_vel_norm, joints
    else:
        return df_vel_norm

def import_prep_ecog(session, monkey=None, **kwargs):
    """
    导入预处理后的 ECoG 数据 (FIF 格式)。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符。
    **kwargs
        转发给 mne.io.read_raw_fif 的额外参数。

    Returns
    -------
    mne.io.Raw
        预处理后的 ECoG 数据（未预加载）。
    """
    from mne.io import read_raw_fif
    sess_str = _sess_str(session, monkey)
    raw = read_raw_fif(path['ecog'] / f"ecog_{sess_str}_prep.fif", preload=False, **kwargs)
    return raw

def import_coor_selection(session, monkey=None):
    """
    导入坐标选择掩码。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符。

    Returns
    -------
    numpy.ndarray
        布尔或整数索引数组。
    """
    sess_str = _sess_str(session, monkey)
    selection = np.load(path['coor'] / f'coor_selection_{sess_str}.npy')
    return selection

def import_euler_angle(session, monkey=None):
    """
    导入欧拉角数据。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符。

    Returns
    -------
    pandas.DataFrame
        欧拉角数据，双层列索引 (joint, dim)。
    """
    sess_str = _sess_str(session, monkey)
    df_euler = pd.read_csv(path['coor'] / f'euler_angle_{sess_str}.csv', header=[0, 1], index_col=0)
    return df_euler

def import_ethogram(session, monkey=None):
    """
    导入行为谱 (ethogram) 数据。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str, optional
        动物标识符。

    Returns
    -------
    pandas.DataFrame
        行为谱数据。
    """
    sess_str = _sess_str(session, monkey)
    df_ethogram = pd.read_csv(path['videos'] / f'ethogram_{sess_str}.csv')
    return df_ethogram

def import_pc(session, postfix='_cartesian_vel_norm'):
    """
    导入主成分 (PC) 数据。

    Parameters
    ----------
    session : str
        会话标识符。
    postfix : str
        文件名后缀，默认 '_cartesian_vel_norm'。

    Returns
    -------
    pandas.DataFrame
        主成分数据。
    """
    return pd.read_csv(path['decomp'] / f'pc_{session}{postfix}.csv', index_col=[0])

def _sess_str(session, monkey):
    """
    构造会话字符串。

    Parameters
    ----------
    session : str
        会话标识符。
    monkey : str or None
        动物标识符。若提供，格式为 'monkey{monkey}_sess{session}'；否则直接返回 session。

    Returns
    -------
    str
        会话字符串。
    """
    s = f'monkey{monkey}_sess{session}' if monkey is not None else session
    return s