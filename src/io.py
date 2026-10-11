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
    将sn映射到对应的location
    参数:
        sn: 单个sn值或sn列表/数组
    返回:
        对应的location值或location列表/数组
    """
    df = import_camera_info()
    df = df.set_index('sn')
    return df.loc[sn, 'location']

def loc2sn(loc):
    """
    将location映射到对应的sn
    参数:
        sn: 单个sn值或sn列表/数组
    返回:
        对应的location值或location列表/数组
    """
    df = import_camera_info()
    df = df.set_index('location')
    return df.loc[loc, 'sn']

def check_tag(tag):
    if tag != "" and tag[0] != "_":
        tag = "_" + tag

    return tag


def import_camera_info():
    # 指定sn列为str类型
    df = pd.read_excel(path['config'] / 'camera_info.xlsx', index_col=0, dtype={'sn': str}) 
    return df

def import_fps(session, monkey=None, stage='quan'):
    """
    导入帧率
    参数:
        session: 会话值
        monkey: 动物值
        stage: 计算帧率的阶段
        |'pos' |'quan' | 'update' | 'setting'
    返回:
        帧率
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
    导入合并后的帧
    参数:
        session: 会话值
        monkey: 动物值
    返回:
        合并后的帧数据框
    """
    fname_aligned = path['raw'] / session / f'aligned_frames.csv'
    df_aligned = pd.read_csv(fname_aligned, index_col=0).rename(columns=sn2loc)
    return df_aligned

def import_aligned_timestamps(session, monkey=None):
    """
    导入合并后的时间戳
    参数:
        session: 会话值
        monkey: 动物值
    返回:
        合并后的时间戳数据框
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
    df = pd.read_excel(path['config'] / 'sess_info.xlsx', index_col=0)

    if if_select:
        df = df.query('selected == 1')

    if session is not None:
        df = df.loc[session]

    return df

def import_montage():
    from mne.channels import read_dig_fif
    montage = read_dig_fif(path['config'] / r'mne_layout_ecog64_20250606-dig.fif')
    return montage

def get_bdf_start_time(path_bdf):
    from pyedflib import EdfReader

    with EdfReader(str(path_bdf)) as f:
        # 获取开始时间（datetime 对象）
        start_time = f.getStartdatetime()  # 返回 datetime.datetime
        start_time = start_time.replace(microsecond=start_time.microsecond * 10) # 似乎解析规则不一样，所以这里要乘以10
    return start_time

def import_ch_names():
    from mne.channels import read_dig_fif
    montage = read_dig_fif(path['config'] / 'mne_layout_ecog64_20250606-dig.fif')
    ch_names = montage.ch_names
    ch_names = [ch for ch in ch_names if '11' not in ch]

    return ch_names

def import_elec_loc(postfix='_64_20250606'):
    df_elec_loc = pd.read_excel(path['config'] / f'elec_loc{postfix}.xlsx', index_col=0)

    return df_elec_loc

def import_config_skeleton_file():
    from yaml import safe_load
    path_yaml = path['config'] / "config_skeleton.yaml"
    with open(path_yaml, 'r', encoding='utf-8') as f:
        config = safe_load(f)    
        return config
    
def import_joints(categories=True):
    config = import_config_skeleton_file()
    joints = config['joints']
    if categories:
        joints = pd.Categorical(joints, categories=joints, ordered=True)
    
    return joints

def import_skeleton_config():
    config = import_config_skeleton_file()
    edges = config['skeleton']
    return edges

def import_annots(session=None, raw=None):
    if session is not None:
        raw = import_ecog_prep(session)
        annots = raw.annotations
    elif raw is not None:
        annots = raw.annotations
    elif annots is None:
        raise ValueError('Please specify either session or raw')
    
    return annots

def import_onsets(session=None, raw=None, annots=None):
    if (session is not None) or (raw is not None):
        annots = import_annots(session=session, raw=raw)

    elif annots is None:
        raise ValueError('Please specify either session or raw')
    
    onsets = annots.to_data_frame(time_format=None).set_index('description')['onset']
    
    return onsets

def import_ecog_prep(session, monkey=None, postfix='_prep', **kwargs):
    
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
    fname = path['ecog'] / f'annots_{session}_bad.csv'
    df = pd.read_csv(fname)
    if not df.query('Annotation == "sample_start"').empty:
        sample_start = df.loc[df['Annotation'] == 'sample_start']['Onset'].values
        df['Onset'] -= sample_start
    else:
        Warning('No time of sample start')

    df = df[df['Annotation'].str.contains('bad', case=False, na=False, regex=False)]

    return df

def import_miss(session, monkey=None, query=None, pad=0):
    """Import bad_miss intervals from the coordinate table.

    Parameters
    ----------
    session : str
        Session identifier, e.g. 'A0' or 'B1'.
    monkey : ignored
        Kept for signature compatibility.

    Returns
    -------
    pandas.DataFrame
        Columns ['Onset', 'Duration', 'Joint'].
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
    """Convert bad_miss table into MNE Annotations.

    Parameters
    ----------
    session : str
        Session identifier.
    joint : str, optional
        If provided, only load bad_miss segments for this joint.
    description : str
        Annotation description prefix.

    Returns
    -------
    mne.Annotations
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
    sess_str = _sess_str(session, monkey)
    pos_file = path['coor'] / f'pos_{sess_str}{postfix}.csv'
    df_pos = pd.read_csv(pos_file, header=[0, 1], index_col=0)

    return df_pos

def import_coor(session, monkey=None, postfix="_standardized"):
    sess_str = _sess_str(session, monkey)
    coor_file = path['coor'] / f'coor_{sess_str}{postfix}.csv'
    df_coor = pd.read_csv(coor_file, header=[0, 1], index_col=0)

    return df_coor

def import_cartesian_quan(session, monkey=None, quans=None):
    sess_str = _sess_str(session, monkey)
    df_quan = pd.read_csv(path['coor'] / f'cartesian_quan_{sess_str}.csv', index_col=0, header=[0, 1])
    joints = df_quan.columns.get_level_values(0).unique().to_list()
    if quans is None:
        quans = df_quan.columns.get_level_values(1).unique().to_list()
    else:
        df_quan = df_quan.xs(quans, axis=1, level=1)
    return df_quan, joints, quans

def import_euler_angle_quan(session, monkey=None, quans=None):
    sess_str = _sess_str(session, monkey)
    df_quan = pd.read_csv(path['coor'] / f'euler_angle_quan_{sess_str}.csv', index_col=0, header=[0, 1])
    joints = df_quan.columns.get_level_values(0).unique().to_list()
    if quans is None:
        quans = df_quan.columns.get_level_values(1).unique().to_list()
    else:
        df_quan = df_quan.xs(quans, axis=1, level=1)
    return df_quan, joints, quans

def import_cartesian_vel_norm(session, monkey=None, postfix="", return_joints=False):
    sess_str = _sess_str(session, monkey)
    df_vel_norm = pd.read_csv(path['coor'] / f'cartesian_vel_norm_{sess_str}{postfix}.csv', index_col=0)
    # joints = df_vel_norm.columns.get_level_values(0).unique().to_list()
    if return_joints:
        joints = import_joints()
        return df_vel_norm, joints
    else:
        return df_vel_norm

def import_angle_vel(session, monkey=None, prefix="angle_vel", postfix="_smoothed", return_joints=False):
    sess_str = _sess_str(session, monkey)
    df_vel_norm = pd.read_csv(path['coor'] / f'{prefix}_{sess_str}{postfix}.csv', index_col=0)
    # joints = df_vel_norm.columns.get_level_values(0).unique().to_list()
    if return_joints:
        joints = import_joints()
        return df_vel_norm, joints
    else:
        return df_vel_norm

def import_kin_quan(session, monkey=None, quan="cartesian_vel_norm", postfix="_smoothed", return_joints=False):
    sess_str = _sess_str(session, monkey)
    df_vel_norm = pd.read_csv(path['coor'] / f'{quan}_{sess_str}{postfix}.csv', index_col=0)
    # joints = df_vel_norm.columns.get_level_values(0).unique().to_list()
    if return_joints:
        joints = import_joints()
        return df_vel_norm, joints
    else:
        return df_vel_norm

def import_prep_ecog(session, monkey=None, **kwargs):
    from mne.io import read_raw_fif
    sess_str = _sess_str(session, monkey)
    raw = read_raw_fif(path['ecog'] / f"ecog_{sess_str}_prep.fif", preload=False, **kwargs)
    return raw

def import_coor_selection(session, monkey=None):
    sess_str = _sess_str(session, monkey)
    selection = np.load(path['coor'] / f'coor_selection_{sess_str}.npy')
    return selection

def import_euler_angle(session, monkey=None):
    sess_str = _sess_str(session, monkey)
    df_euler = pd.read_csv(path['coor'] / f'euler_angle_{sess_str}.csv', header=[0, 1], index_col=0)
    return df_euler

def import_ethogram(session, monkey=None):
    sess_str = _sess_str(session, monkey)
    df_ethogram = pd.read_csv(path['videos'] / f'ethogram_{sess_str}.csv')
    return df_ethogram

def import_pc(session, postfix='_cartesian_vel_norm'):
    return pd.read_csv(path['decomp'] / f'pc_{session}{postfix}.csv', index_col=[0])

def _sess_str(session, monkey):
    s = f'monkey{monkey}_sess{session}' if monkey is not None else session
    return s