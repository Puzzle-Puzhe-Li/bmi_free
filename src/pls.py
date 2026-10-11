from pathlib import Path
import numpy as np
import pandas as pd
import datetime

from scipy.stats import zscore, ttest_rel, pearsonr

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import numpy as np
from joblib import Parallel, delayed, dump, load as jl_load

from bmi_free.io import (
    import_annots, sessions, import_session_info, import_joints, import_fps,
    path as data_path, import_onsets, get_bdf_start_time, import_ecog_prep, import_miss_annots
)
from bmi_free.features import (
    select_coors, select_features, select_features_and_coors,
    shuffle_X
)
from bmi_free.plot import palette
from bmi_free.utils import Tag, p2aster
import matplotlib.pyplot as plt, seaborn as sns
from datetime import timedelta, timezone

import mne
import inflection

def simp_pls_python(
    X: np.ndarray,
    y: np.ndarray,
    ncomp: int = 50,
    train_frac: float = 0.7,
    scale_X: bool = True,   # 建议保持 True，除非确认 simp_pls.m 未标准化
    scale_y: bool = False,
) -> tuple:
    """
    simp_pls 的 Python 纯复现版本（无需 MATLAB）。
    
    Parameters
    ----------
    X : (T, nchan, ...) 输入特征
    y : (T, 1) 或 (T,) 目标变量
    ncomp : PLS 成分数
    train_frac : 训练集比例（前段）
    scale_X : 是否对 X 做 z-score 标准化（用训练集统计量）
    scale_y : 是否对 y 做 z-score 标准化
    
    Returns
    -------
    b : 回归系数，shape 与 X[0] 相同
    r : 测试集 Pearson 相关系数
    """
    orig_shape = X.shape[1:]
    T_total = X.shape[0]
    X_2d = X.reshape(T_total, -1).astype('float64')
    y    = y.reshape(-1, 1).astype('float64')
    
    # 训练/测试划分（时间顺序）
    n_train = int(T_total * train_frac)
    X_tr, X_te = X_2d[:n_train], X_2d[n_train:]
    y_tr, y_te = y[:n_train],    y[n_train:]
    
    # 可选：X 标准化
    sx = None
    if scale_X:
        sx = StandardScaler()
        X_tr = sx.fit_transform(X_tr)
        X_te = sx.transform(X_te)
    
    # 可选：y 标准化
    sy = None
    if scale_y:
        sy = StandardScaler()
        y_tr = sy.fit_transform(y_tr)
    
    # PLS 拟合
    pls = PLSRegression(n_components=ncomp, scale=False, max_iter=500)
    pls.fit(X_tr, y_tr)
    
    # 预测
    y_pred = pls.predict(X_te)
    if scale_y:
        y_pred = sy.inverse_transform(y_pred)
    
    r, _ = pearsonr(y_te.flatten(), y_pred.flatten())
    b = pls.coef_.flatten().reshape(orig_shape)
    
    return b, float(r)    

def zscore_feature(X):
    # 假设 X_freq_time shape: (frame, channel, time, freq)
    n_frame, n_channel, n_time, n_freq = X.shape

    # 重塑为 (channel, freq, frame * time)，将每个通道每个频率点的所有时间点串联
    X = X.transpose(1, 3, 0, 2).reshape(n_channel, n_freq, -1)

    # 沿着时间轴进行 zscore 标准化，忽略 NaN
    X = zscore(X, axis=-1, nan_policy='omit')

    # 还原为原始形状 (frame, channel, time, frequency)
    X = X.reshape(n_channel, n_freq, n_frame, n_time).transpose(2, 0, 3, 1)
    return X

def build_full_time_axis(
    n_frames_total: int,
    alive_indices: np.ndarray,
    n_train: int,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
    y_valid: np.ndarray,
    y_all: np.ndarray,
) -> dict:
    """
    构建完整时间轴的 y_true, y_pred
    
    Parameters
    ----------
    n_frames_total : int
        原始总帧数（经过 selection 之前）
    alive_indices : np.ndarray
        经过 selection 后保留的帧索引 (在原始帧坐标中)
    n_train : int
        训练集帧数（在 alive_indices 坐标系中）
    y_train_pred : np.ndarray, shape (n_train,)
        训练集预测值
    y_test_pred : np.ndarray, shape (n_frames_total - n_train,)
        测试集预测值
    y_valid : np.ndarray, shape (len(alive_indices),)
        所有存活帧的真实值（训练集 + 测试集，按 alive_indices 顺序）        
    y_all : np.ndarray, shape (n_frames_total,)
        所有原值
    
    Returns
    -------
    dict
        {
            'y_true_full': (n_frames_total,)  存活帧有值，缺失帧 NaN
            'y_pred_full': (n_frames_total,)  同上
            'valid_mask': (n_frames_total,)   True 表示该帧存活
        }
    """
    # y_true_full = np.full(n_frames_total, np.nan, dtype=np.float64)
    y_pred_full = np.full(n_frames_total, np.nan, dtype=np.float64)
    
    # 拼接训练集和测试集的预测值 (按 alive_indices 的顺序)
    y_pred_all = np.concatenate([y_train_pred.flatten(), y_test_pred.flatten()])
    
    # y_true_full[alive_indices] = y_valid.flatten()
    y_pred_full[alive_indices] = y_pred_all
    
    valid_mask = np.zeros(n_frames_total, dtype=bool)
    valid_mask[alive_indices] = True
    
    return {
        # 'y_true_full': y_true_full,
        'y_true_full': y_all,
        'y_pred_full': y_pred_full,
        'valid_mask': valid_mask,
    }


def sliding_r_with_valid_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_mask: np.ndarray,
    window: int,
    step: int,
) -> np.ndarray:
    """
    只在窗口内所有帧都 valid 时计算 r，否则输出 NaN
    
    Parameters
    ----------
    y_true : np.ndarray, shape (n_frames,)
        完整时间轴的真实值
    y_pred : np.ndarray, shape (n_frames,)
        完整时间轴的预测值
    valid_mask : np.ndarray, shape (n_frames,)
        True 表示该帧有效
    window : int
        滑动窗口大小（帧数）
    step : int
        滑动步长（帧数）
    
    Returns
    -------
    r_ts : np.ndarray, shape (n_frames,)
        每个时间点的滑动窗口 r，无效窗口填充 NaN
    """
    n = len(y_true)
    
    # 如果窗口大于总长度，返回全 NaN
    if window > n:
        return np.full(n, np.nan)
    
    # starts = np.arange(0, n - window + 1, step)
    starts = np.arange(0, n - window + 1)
    centers = starts + window // 2
    r_values = np.full(len(starts), np.nan)
    
    for i, start in enumerate(starts):
        idx = slice(start, start + window)
        
        # 检查窗口内所有帧是否都有效，且没有 NaN
        if valid_mask[idx].all() and not np.isnan(y_true[idx]).any() and not np.isnan(y_pred[idx]).any():
            r_values[i] = pearsonr(y_true[idx], y_pred[idx])[0]
    
    # 插值回每个时间点，边缘用 NaN 填充
    if len(starts) == 1:
        return np.full(n, r_values[0])
    
    r_ts = np.full(n, np.nan)
    r_ts[centers] = r_values
    
    return r_ts


def get_annotations_for_edf(
    valid_mask: np.ndarray,
    n_train: int,
    sample_start: float,
    sfreq: float,
    raw_annotations: mne.Annotations = None,
    alive_indices: np.ndarray = None,
) -> mne.Annotations:
    """
    生成导出 EDF 的 annotations
    
    Parameters
    ----------
    valid_mask : np.ndarray, shape (n_frames_total,)
        True 表示该帧有效（存活），False 表示缺失
    n_train : int
        训练集帧数（在 alive_indices 坐标系中）
    sample_start : float
        sample_start 时间（秒）
    sfreq : float
        帧率 (Hz)
    raw_annotations : mne.Annotations, optional
        原始 ECoG 的 annotations，用于继承 BAD 片段
    alive_indices : np.ndarray, optional
        存活帧的原始索引，用于将帧索引转换为时间
    
    Returns
    -------
    mne.Annotations
        包含所有标注的 Annotations 对象
    """
    onset_list = []
    duration_list = []
    description_list = []
    
    # 1. 继承原始 BAD 标注
    if raw_annotations is not None:
        for onset, duration, desc in zip(
            raw_annotations.onset,
            raw_annotations.duration,
            raw_annotations.description
        ):
            if desc.lower().startswith('bad'):
                # 转换为相对于 sample_start 的时间
                onset_list.append(onset - sample_start)
                duration_list.append(duration)
                description_list.append(desc)
    
    # 2. 标注 train/test 分割点
    if alive_indices is not None and n_train < len(alive_indices):
        split_frame_idx = alive_indices[n_train]  # 测试集第一帧的原始索引
        split_time = split_frame_idx / sfreq      # 相对 sample_start 的时间
        onset_list.append(split_time)
        duration_list.append(0.0)
        description_list.append('TRAIN_TEST_SPLIT')
    
    # 3. 标注 selection 缺失的连续片段 (valid_mask 中 False 的连续段)
    if alive_indices is not None:
        # 将 alive_indices 转换为 set 以便快速查找
        alive_set = set(alive_indices)
        n_total = len(valid_mask)
        
        # 找到所有缺失帧的连续区间
        in_gap = False
        gap_start = 0
        
        for i in range(n_total + 1):  # +1 是为了处理末尾的 gap
            if i < n_total and i not in alive_set:
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    # 结束一个 gap
                    gap_end = i  # 开区间 [gap_start, gap_end)
                    onset_time = gap_start / sfreq
                    duration = (gap_end - gap_start) / sfreq
                    if duration > 0:
                        onset_list.append(onset_time)
                        duration_list.append(duration)
                        description_list.append('BAD_SELECTION')
                    in_gap = False
    
    # 4. 如果没有任何标注，添加一个占位标注 (mne 要求非空)
    if len(onset_list) == 0:
        onset_list.append(0.0)
        duration_list.append(0.0)
        description_list.append('NO_ANNOTATIONS')
    
    return mne.Annotations(
        onset=np.array(onset_list),
        duration=np.array(duration_list),
        description=np.array(description_list),
        orig_time=0.0,  # 相对于 sample_start
    )


def export_pred(
    edf_data: dict,
    path_edf: Path,
    sfreq: float,
    n_frames_total: int,
    start_time: datetime.datetime = None,
    annots: mne.Annotations = None,
    window_s: float = 10.0,
    overlap: float = 0.2,
    joint_order: list = None,
) -> None:
    """
    导出完整时间轴的预测结果 EDF
    
    Parameters
    ----------
    edf_data : dict
        每个 joint 包含 y_true_full, y_pred_full, valid_mask, r_ts
        由 build_full_time_axis 和 sliding_r_with_valid_mask 生成
    path_edf : Path
        输出文件路径
    sfreq : float
        采样率 (Hz)
    n_frames_total : int
        总帧数
    alive_indices : np.ndarray
        存活帧的原始索引
    n_train : int
        训练集帧数
    # sample_start : float
    #     sample_start 时间（秒），用于 annotations 的时间偏移
    start_time : datetime, optional
        EDF 起始真实时间
    raw_annotations : mne.Annotations, optional
        原始 ECoG annotations，用于继承 BAD 标注
    window_s : float
        滑动窗口长度（秒）
    overlap : float
        窗口重叠比例 [0, 1)
    joint_order : list, optional
        指定 joint 的导出顺序
    """
    window = int(window_s * sfreq)
    step = max(1, int(window * (1 - overlap)))
    joints = edf_data.keys()
    
    ch_names, ch_types, data_rows = [], [], []
    
    for joint in joints:
        d = edf_data[joint]
        y_true = d['y_true_full']
        y_pred = d['y_pred_full']
        r_ts = d['r_ts']
        valid_mask = d['valid_mask']
        
        # 将 NaN 转换为 EDF 可用的值 (用0表示NaN)
        y_true_export = np.nan_to_num(y_true, nan=0)
        y_pred_export = np.nan_to_num(y_pred, nan=0)
        r_export = np.nan_to_num(r_ts, nan=0)

        ch_names.extend([f'{joint}', f'{joint}_pred', f'{joint}_r'])
        ch_types.extend(['misc', 'misc', 'misc'])
        data_rows.extend([y_true_export*1.5, y_pred_export*1.5 + 150, r_export * 150])
    
    # 添加 train/test 标记通道

    
    data_array = np.array(data_rows, dtype=np.float64)
    
    # 创建 info
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types,
        verbose='CRITICAL'
    )
    
    if start_time is not None:
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        info.set_meas_date(start_time)
    
    raw = mne.io.RawArray(data_array, info, verbose='CRITICAL')
    raw.set_annotations(annots, verbose='CRITICAL')
    
    # 导出
    raw.export(path_edf, overwrite=True, verbose='CRITICAL')
    print(f'✅ Exported to {path_edf}')


def pls_joint(
    features,
    df,
    joint,
    path_out,
    tag="",
    ncomp=10,
    train_frac=0.7,
    scale_X=False,
    scale_y=False,
    save_items=['r_test'],
    feature_selection=None,
    if_export=False,
    shuffle_dims=None,
    involvement=None
):
    """Core per-joint PLS workflow used by pls_joints.

    Parameters
    ----------
    features : mne.time_frequency.EpochsTFRArray
        Full feature matrix.
    df : pandas.DataFrame
        Velocity dataframe for all joints.
    joint : str
        Joint name to train.
    path_out : Path
        Output directory for model/weight files.
    tag : str
        Optional filename suffix.
    ncomp : int
        Number of PLS components.
    train_frac : float
        Training split fraction.
    scale_X : bool
        Whether to standardize X.
    scale_y : bool
        Whether to standardize y.
    save_items : list
        Which items to save into the model payload.

    Returns
    -------
    dict
        Result dictionary for this joint.
    """

    session = features.info['description']
    fname_model = path_out / f'model_{session}_{joint}{tag}.joblib'
    fname_weights = path_out / f'weights_{session}_{joint}{tag}.h5'    

    if fname_model.exists():
        if if_export:   # 只有导出才需要返回结果
            saved = jl_load(fname_model)
            pls = saved.get('pls')
            sy = saved.get('sy', None)

            if pls is not None:
                y_pred_train = pls.predict(X_tr)
                y_pred_test = pls.predict(X_te)
                if sy is not None:
                    y_pred_train = sy.inverse_transform(y_pred_train)
                    y_pred_test = sy.inverse_transform(y_pred_test)

                r_train, _ = pearsonr(y_train.flatten(), y_pred_train.flatten())
                r_test, _ = pearsonr(y_test.flatten(), y_pred_test.flatten())
                r_all, _ = pearsonr(
                    np.concatenate([y_train, y_test]),
                    np.concatenate([y_pred_train.flatten(), y_pred_test.flatten()]),
                )

                return {
                    'joint': joint,
                    'r_train': float(r_train),
                    'r_test': float(r_test),
                    'r_all': float(r_all),
                    'y_train_pred': y_pred_train.flatten(),
                    'y_test_pred': y_pred_test.flatten(),
                    'y_train': y_train,
                    'y_test': y_test,
                    'pls': pls,
                    'sy': sy,
                    'loaded': True,
                    'alive_indices': features_joint.selection,
                    'n_train': n_train,
                    't_test': t_test
                }
        else: pass

    else:

        frame_times = features.metadata['frame_times']
        coor_selection = select_coors(session, joint, frame_times, involvement=involvement)
        features_joint, df_vel_joint = select_features_and_coors(
            features,
            df,
            feature_selection=feature_selection,
            coor_selection=coor_selection
        )
        X = features_joint.get_data(picks='all')
        T = X.shape[0]
        orig_shape = X.shape[1:]

        if shuffle_dims is not None:
            X = shuffle_X(X, shuffle_dims=shuffle_dims, seed=98, 
                          save_as=path_out / f'shuffle_{session}_{joint}{tag}.pkl' )

        X = X.reshape(T, -1)

        n_train = int(T * train_frac)
        X_tr = X[:n_train]
        X_te = X[n_train:]

        if scale_X:
            sx = StandardScaler()
            X_tr = sx.fit_transform(X_tr)
            X_te = sx.transform(X_te)

        y = df_vel_joint[joint].values.astype('float64')
        y_train = y[:n_train]
        y_test = y[n_train:]
        t_test = df_vel_joint.index[n_train]

        y_tr = y_train.reshape(-1, 1)
        sy = None
        if scale_y:
            sy = StandardScaler()
            y_tr = sy.fit_transform(y_tr)

        pls = PLSRegression(n_components=ncomp, scale=False, max_iter=500)
        pls.fit(X_tr, y_tr)

        y_pred_train = pls.predict(X_tr)
        y_pred_test = pls.predict(X_te)

        if sy is not None:
            y_pred_train = sy.inverse_transform(y_pred_train)
            y_pred_test = sy.inverse_transform(y_pred_test)

        r_train, _ = pearsonr(y_train.flatten(), y_pred_train.flatten())
        r_test, _ = pearsonr(y_test.flatten(), y_pred_test.flatten())
        r_all, _ = pearsonr(
            np.concatenate([y_train, y_test]),
            np.concatenate([y_pred_train.flatten(), y_pred_test.flatten()]),
        )

        if 'weights' in save_items:
            save_items.remove('weights')
            W = pls.coef_.flatten().reshape(orig_shape).astype('float64')
            if W.ndim == 2:
                W = W[:, :, np.newaxis]
                w_times = np.array([0.0])
            else:
                w_times = features_joint.times

            weights = mne.time_frequency.RawTFRArray(
                info=features_joint.info,
                data=W,
                times=w_times,
                freqs=features_joint.freqs,
                method='PLS_weight',
            )
            weights.save(fname_weights, overwrite=True)

        model_payload = {item: eval(item) for item in save_items}
        model_payload.update({'r_test': float(r_test)})

        if if_export:
            model_payload.update({
                'pls': pls,
                'sy': sy,
                'r_train': float(r_train),
                'r_all': float(r_all),
            })
        dump(model_payload, fname_model)

        return {
            'joint': joint,
            'r_train': float(r_train),
            'r_test': float(r_test),
            'r_all': float(r_all),
            'y_train_pred': y_pred_train.flatten(),
            'y_test_pred': y_pred_test.flatten(),
            'y_train': y_train,
            'y_test': y_test,
            'pls': pls,
            'sy': sy,
            'loaded': False,
            'alive_indices': features_joint.selection,
            'n_train': n_train,
            't_test': t_test
        }


def pls_joints(
    features,
    df,
    path_out,
    tag="",
    n_jobs=3,
    verbose=0,
    ncomp=10,
    train_frac=0.7,
    scale_X=False,
    scale_y=False,
    shuffle_dims=None,
    involvement=None,
    if_export=False,
    if_log_r=False,
    save_items=['r_test'],
    vel_thr=None,
    window_s=2.0,
    overlap=0.2,
):

    session = features.info['description']
    sfreq = features.info['sfreq']
    joints = df.columns.tolist()

    n_frames_total = len(features)

    feature_selection = select_features(features)


    if n_jobs > 0:
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(pls_joint)(
                features=features,
                df=df,
                joint=joint,
                path_out=path_out,
                tag=tag,
                ncomp=ncomp,
                train_frac=train_frac,
                scale_X=scale_X,
                scale_y=scale_y,
                save_items=save_items,
                feature_selection=feature_selection,
                if_export=if_export,
                shuffle_dims=shuffle_dims,
                involvement=involvement
            ) for joint in joints
        )
    else:
        results = [
            pls_joint(
                features=features,
                df=df,
                joint=joint,
                path_out=path_out,
                tag=tag,
                ncomp=ncomp,
                train_frac=train_frac,
                scale_X=scale_X,
                scale_y=scale_y,
                save_items=save_items,
                feature_selection=feature_selection,
                if_export=if_export,
                shuffle_dims=shuffle_dims,
                involvement=involvement
            ) for joint in joints
        ]

    # 导出 EDF
    if if_export:
        fname_edf = path_out / f'pred_{session}{tag}.edf'
        if not fname_edf.exists():
            # 汇总结果并构建 edf_data
            all_edf_data = {}
            all_y_train = []
            all_y_train_pred = []
            all_y_test = []
            all_y_test_pred = []
            all_t_test = []
            
            for res in results:
                joint = res['joint']
                
                if if_log_r:
                    print(f'  {joint}: train_r={res["r_train"]:.4f}, test_r={res["r_test"]:.4f}, all_r={res["r_all"]:.4f}')
                
                # 构建完整时间轴
                y_valid = np.concatenate([res['y_train'], res['y_test']])
                y_pred_all = np.concatenate([res['y_train_pred'], res['y_test_pred']])
                
                full_data = build_full_time_axis(
                    n_frames_total=n_frames_total,
                    alive_indices=res['alive_indices'],
                    n_train=res['n_train'],
                    y_train_pred=res['y_train_pred'],
                    y_test_pred=res['y_test_pred'],
                    y_valid=y_valid,
                    y_all=df[joint].values
                )
                
                # 计算滑动窗口 r
                window = int(window_s * sfreq)
                step = max(1, int(window * (1 - overlap)))
                r_ts = sliding_r_with_valid_mask(
                    y_true=full_data['y_true_full'],
                    y_pred=full_data['y_pred_full'],
                    valid_mask=full_data['valid_mask'],
                    window=window,
                    step=step,
                )
                
                full_data['r_ts'] = r_ts
                all_edf_data[joint] = full_data
                
                # 收集用于统计的 y
                all_y_train.append(res['y_train'])
                all_y_train_pred.append(res['y_train_pred'])
                all_y_test.append(res['y_test'])
                all_y_test_pred.append(res['y_test_pred'])
                all_t_test.append(res['t_test'])
    
            try:
                sample_start = import_onsets(session=session)['sample_start']
                fname_bdf = data_path['ecog'] / f'ecog_{session}_prep.bdf'
                bdf_start_time = get_bdf_start_time(fname_bdf)
                start_time = bdf_start_time + timedelta(seconds=float(sample_start))
                start_time = start_time.replace(tzinfo=datetime.timezone.utc)
            except Exception as e:
                print(f'无法获取真实起始时间: {e}')
                start_time = None
                sample_start = 0.0            
            
            # 获取原始 annotations
            raw_annotations = import_annots(session=session)
            
            # 将 annotations 的 onset 转换为相对于 raw 的时间（秒）
            raw_annotations = mne.Annotations(
                onset=raw_annotations.onset - sample_start,
                duration=raw_annotations.duration,
                description=raw_annotations.description,
                orig_time=start_time,
            )
            miss_annots = import_miss_annots(session=session, orig_time=start_time)

            if 'Wrist(L)' not in joints:
                jsl_map = {
                    'Nose':['Neck'],
                    'Head':['Neck', 'Back'],
                    'Neck':['Neck', 'Back'],
                    'Shoulder(L)':['Back', 'Shoulder(L)', 'Elbow(L)'],
                    'Shoulder(R)':['Back', 'Shoulder(R)', 'Elbow(R)'],
                    'Elbow(L)':['Shoulder(L)', 'Elbow(L)'],
                    'Elbow(R)':['Shoulder(R)', 'Elbow(R)'],
                    'Wrist(L)':['Shoulder(L)', 'Elbow(L)'],
                    'Wrist(R)':['Shoulder(R)', 'Elbow(R)'],
                    'Back':['Neck', 'Back', 'Shoulder(L)', 'Shoulder(R)'],
                    'Sacrum':['Back', 'Sacrum', 'Thigh(L)', 'Thigh(R)'],
                    'Thigh(L)':['Sacrum', 'Thigh(L)', 'Knee(L)'],
                    'Thigh(R)':['Sacrum', 'Thigh(R)', 'Knee(R)'],
                    'Knee(L)':['Thigh(L)', 'Knee(L)'],
                    'Knee(R)':['Thigh(R)', 'Knee(R)'],
                    'Ankle(L)':['Thigh(L)','Knee(L)'],
                    'Ankle(R)':['Thigh(R)','Knee(R)'],
                }
                miss_annots = mne.Annotations(
                    onset=miss_annots.onset,
                    duration=miss_annots.duration,
                    description=miss_annots.description, 
                    ch_names=[jsl_map[j[0]] for j in miss_annots.ch_names],
                    orig_time=start_time)          
            
            test_annots = mne.Annotations(
                onset=np.array(all_t_test),
                duration=0,
                description='test_set',
                ch_names=[[j] for j in joints],
                orig_time=start_time,
            )
            
            annots = raw_annotations + miss_annots + test_annots
            
            export_pred(
                edf_data=all_edf_data,
                path_edf=fname_edf,
                sfreq=sfreq,
                n_frames_total=n_frames_total,
                start_time=start_time,
                annots=annots,
                window_s=window_s,
                overlap=overlap,
                joint_order=joints,
            )
    

def load_weights(path_out, tag="", joints=None, absolute=False, baseline=(None, None), mode='zscore', session=None):

    if joints is None:
        joints = import_joints()
        
    weights_dict = {}
    sessions_loaded = [session] if session is not None else sessions
    for session in sessions_loaded:        
        W = []
        joints_include = []
        for joint in joints:
            fname = path_out / f'weights_{session}_{joint}{tag}.h5'
            if fname.exists():
                joints_include.append(joint)
                weights = mne.time_frequency.read_tfrs(fname)
                W.append(weights.get_data())
               

        W = np.stack(W) 
        
        if absolute:
            W = np.abs(W)                     
        
        weights = mne.time_frequency.EpochsTFRArray(
            info=weights.info,
            data=W,                     # (n_epochs, n_channels, n_freqs, n_times)
            times=weights.times,
            freqs=weights.freqs,
            metadata=pd.DataFrame({'joint': joints_include}),
        )  
        
        if baseline is not None:
            weights.apply_baseline(baseline, mode=mode)
        
        weights_dict[session] = weights

    if session is not None:
        return weights_dict[session]
    return weights_dict

def load_r(path_out, joints=None, sort_index=True):
    from joblib import load as jl_load
    import re

    if joints is None:
        joints = import_joints()

    records = []

    for session in sessions:
        monkey = session[:1]
        for joint in joints:
            for fname_model in sorted(path_out.glob(f'model_{session}_{joint}*.joblib')):
                tag = Tag.from_str(fname_model.stem.replace(f'model_{session}_{joint}', ''))
                records.append({
                    'monkey': monkey, 'joint': joint, 
                    'r': jl_load(fname_model)['r_test']
                    } | tag)        


    df_r = pd.DataFrame(records)
    df_r['joint'] = pd.Categorical(df_r['joint'], categories=[j for j in joints if j in df_r['joint'].unique()])
    
    if sort_index:
        df_r = df_r.set_index(['monkey', 'joint'])
        df_r = df_r.sort_index()

    return df_r

def _infer_x(data):
    """
    自动推断 x， 如果data里有且仅有r列，则x为"joint"，如果有除了r的另一列，则为这一列；如果除了r还多于或等于两列，则报错
    """
    keys = [i for i in data.columns if i not in ['r', 'monkey', 'joint']]
    if len(keys) == 0:
        return 'joint'
    elif len(keys) == 1:
        return keys[0]
    else:
        return keys[0]
    
def plot_r(data, plot_fun=sns.barplot, rot_x=False, **kwargs):

    df_r = load_r(data) if isinstance(data, (str, Path)) else data

    if 'joint' in df_r.index.names:
        df_r = df_r.reset_index(['joint']).reset_index()
    x = kwargs.pop('x', _infer_x(df_r))
    y = kwargs.pop('y', 'r')
    n_x = df_r[x].unique().size if x is not None else 1

    plot_fun(
        data=df_r,
        x=x,
        y=y,
        hue='monkey',
        palette=palette['monkey'],
        **kwargs
    )
    if kwargs.get('legend', True):
        plt.legend(title='Monkey', ncol=2)

    plt.yticks(minor=True, ticks=np.arange(0, 1, 0.05))
    plt.ylim(0, .8)
    
    if plt.gcf().axes.index(plt.gca()) == 0:
        plt.gcf().set_size_inches(n_x * 0.5, 3)
    if x == 'joint' or rot_x:
        plt.xticks(rotation=30, ha='right')


def stat_r(data, ctrl_group=None):

    df_r = load_r(data) if isinstance(data, (str, Path)) else data

    x = _infer_x(df_r)
    
    df_r.reset_index('joint', inplace=True)
    df_r.set_index(x, append=True, inplace=True)
    df_r_stat = df_r.groupby(['monkey', x])['r'].agg(['mean', 'std']).round(3)
    groups = df_r.index.get_level_values(x).unique().to_list()

    if ctrl_group is not None:

        for session in sessions:
            monkey = session[:1]

            # 计算Low High分别和All组的配对t检验
            for group in groups:
                if group == ctrl_group:
                    continue

                t, p = ttest_rel(
                    df_r.loc[(monkey, group), 'r'],
                    df_r.loc[(monkey, ctrl_group), 'r'],
                )

                df_r_stat.loc[(monkey, group), 't'] = t.round(3)
                df_r_stat.loc[(monkey, group), 'p'] = f'{p:.3e}({p2aster(p)})'
    
    df_r_stat.columns = pd.CategoricalIndex(df_r_stat.columns, categories=df_r_stat.columns)
    df_r_stat = df_r_stat.unstack(level=0).swaplevel(axis=1).sort_index(axis=1)
    return df_r_stat