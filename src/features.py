"""
features.py
-----------
Time-frequency representation (TFR) utilities + feature slicing.

Public functions
----------------
compute_tfr(raw, freqs, method, **kwargs)  →  mne.time_frequency.RawTFRArray
slice_features(tfr, annots, ...)           →  Features
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import pickle

from bmi_free.io import import_onsets, import_ecog_prep, import_miss, import_fps, import_bad, import_involvement
from bmi_free.utils import select_epoch, shuffle

import mne


_THIS_DIR   = Path(__file__).parent.resolve()
_MATLAB_DIR = _THIS_DIR / "matlab"



def compute_tfr(
    raw: np.ndarray,
    freqs: np.ndarray = np.arange(10, 150, 10),
    method: Literal["cwt_matlab", "multitaper", "morlet"] = "multitaper",
    **kwargs,
) -> "mne.time_frequency.RawTFR":
    """Compute a time-frequency representation of multichannel ECoG data.

    Parameters
    ----------
    raw : ndarray, shape (n_channels, n_samples)
        Raw signal, already time-delay corrected.
    freqs : ndarray, shape (n_freq_bins,)
        Centre frequencies of the desired frequency bins (Hz).
        Bin edges are inferred as midpoints between consecutive centres, with
        symmetric half-steps at both ends.
    method : str
        ``'cwt_matlab'`` – Morse CWT via MATLAB engine.
    **kwargs
        Forwarded to the selected backend.

        Required for ``'cwt_matlab'``:
          info      (mne.Info) : channel metadata; ``info['sfreq']`` must match
                                 the effective sampling rate of ``time_out``
          fs        (int)      : sampling rate of *raw* in Hz
          time_in   (ndarray)  : time axis of *raw*, shape (n_samples,)
          time_out  (ndarray)  : target time axis for output, shape (n_out,)
        Optional:
          voices        (int)     : voices-per-octave (default 32)
          freq_bin_edge (ndarray) : override auto-computed bin edges

    Returns
    -------
    mne.time_frequency.RawTFRArray
        ``data`` shape: ``(n_channels, n_freqs, n_times_out)``
    """
    fs_tfr = kwargs.get('fs_tfr', 300)
    def _infer_decim(fs0, fs):
        decim = int(fs0 / fs)
        fs1 = fs * decim
        return decim, fs1

    if method == "cwt_matlab":
        tfr = _compute_tfr_cwt_matlab_isolated(raw, freqs, **kwargs)
    elif method == 'morlet' or method == 'multitaper':
        decim, fs1 = _infer_decim(raw.info['sfreq'], fs_tfr)
        t_window = 0.1
        n_cycles = kwargs.pop('n_cycles', freqs * t_window)
        tfr = raw.resample(fs1).compute_tfr(method, freqs, decim=decim, reject_by_annotation=False, n_cycles=n_cycles, **kwargs)
    else:
        raise ValueError(f"Unknown TFR method: {method!r}. Available: 'cwt_matlab', 'morlet', 'multitaper'.")
    tfr._data = tfr._data.astype(np.float32)
    tfr.info['description'] = raw.info['description']
    return tfr

def _cwt_matlab_worker(raw, freqs, kwargs):
    """子进程入口（loky）：直接 return 结果，不需要 Queue。"""
    result = _compute_tfr_cwt_matlab(raw, freqs, **kwargs)
    return {
        "data":     result.data,
        "times":    result.times,
        "freqs":    result.freqs,
        "ch_names": result.ch_names,
        "sfreq":    result.info["sfreq"],
    }


def _compute_tfr_cwt_matlab_isolated(raw, freqs, **kwargs):
    """在独立子进程中运行 MATLAB CWT，避免 matlab.engine 污染主进程命名空间。
    使用 loky 而非 multiprocessing.Process，兼容 Jupyter Notebook。
    返回的 RawTFRArray 会继承 raw.annotations。
    """
    from loky import get_reusable_executor
    executor = get_reusable_executor(max_workers=1)
    future = executor.submit(_cwt_matlab_worker, raw, freqs, kwargs)
    result = future.result()  # 子进程异常会在此处重新抛出

    info = mne.create_info(
        ch_names=result["ch_names"],
        sfreq=result["sfreq"],
        ch_types=["eeg"] * len(result["ch_names"]),
    )
    times = result["times"] - result["times"][0]    # 为了和 mne.time_frequency.RawTFR一致
    tfr = mne.time_frequency.RawTFRArray(
        info=info,
        data=result["data"],
        times=times,
        freqs=result["freqs"],
        method="cwt_matlab",
    )

    return tfr


def _compute_tfr_cwt_matlab(
    raw: np.ndarray,
    freqs: np.ndarray,
    *,
    fs_tfr: int = 300,
    voices: int = 32,
    freq_bin_edge: np.ndarray | None = None,
) -> "mne.time_frequency.RawTFRArray":
    import matlab.engine

    time_in   = raw.times      

    duration  = time_in[-1] - time_in[0]
    n_out     = int(np.floor(fs_tfr * duration + 1))
    time_out  = np.linspace(time_in[0], (n_out - 1) / fs_tfr + time_in[0], n_out)    

    if freq_bin_edge is None:
        freqs = np.asarray(freqs, dtype=float)
        if len(freqs) < 2:
            raise ValueError("Need at least 2 centre frequencies to infer bin edges.")
        midpoints  = (freqs[:-1] + freqs[1:]) / 2.0
        half_left  = freqs[0]  - (midpoints[0]  - freqs[0])
        half_right = freqs[-1] + (freqs[-1] - midpoints[-1])
        freq_bin_edge = np.concatenate([[half_left], midpoints, [half_right]])

    freq_bin_edge = np.asarray(freq_bin_edge, dtype=float)

    data_mat = matlab.double(raw.get_data().T.tolist())
    edge_mat = matlab.double(freq_bin_edge.reshape(1, -1).tolist())
    tin_mat  = matlab.double(time_in.reshape(-1, 1).tolist())
    tout_mat = matlab.double(time_out.reshape(-1, 1).tolist())
    fs_mat = matlab.double(raw.info['sfreq'])

    eng     = _get_matlab_engine()
    tfr_mat = eng.compute_tfr_cwt(
        data_mat, fs_mat, edge_mat, tin_mat, tout_mat, float(voices), nargout=1
    )
    close_matlab_engine()
    del eng  

    # MATLAB: (n_out, n_channels, n_freqs) → MNE: (n_channels, n_freqs, n_times)
    tfr_np = np.array(tfr_mat).transpose(1, 2, 0).astype(np.float32)

    info = mne.create_info(
        ch_names=raw.ch_names,
        sfreq=fs_tfr,
        ch_types=["eeg"] * len(raw.ch_names)
        )

    return mne.time_frequency.RawTFRArray(
        info=info, data=tfr_np, times=time_out, freqs=freqs, method="cwt_matlab"
    )


# ============================================================================
# MATLAB engine singleton
# ============================================================================

_matlab_engine = None


def _get_matlab_engine():
    global _matlab_engine
    import matlab.engine

    if _matlab_engine is None:
        _matlab_engine = matlab.engine.start_matlab()
        _matlab_engine.addpath(str(_MATLAB_DIR), nargout=0)
    return _matlab_engine


def close_matlab_engine() -> None:
    """Shut down the cached MATLAB engine."""
    global _matlab_engine
    if _matlab_engine is not None:
        _matlab_engine.quit()
        _matlab_engine = None


def slice_features(
    tfr,
    time_range: tuple[float, float] = (-1.0, 0.0),
    time_step: float = 0.1,
    amp_feature=False,
    if_show: bool = False,
    apply_baseline: bool = False,
) -> "mne.timefrequency.EpochTFRArray":
    """从 TFR 对象中按帧切片，组织 (frame, channel, freq, time) 特征矩阵。

    Parameters
    ----------
    tfr : mne.time_frequency.RawTFR
        预计算好的 TFR，``tfr.info['sfreq']`` 为其内部时间分辨率（如 300 Hz）。
    time_range : (t0, t1)
        每帧截取的时间窗口 [t0, t1)，单位秒（相对于帧的"0 时刻"）。
    time_step : float or None
        **特征时间步长**（Hz）。若不为 None 且与 TFR 采样率不一致，则会在
        时间窗口内将连续 ``bin_size = tfr_sfreq * time_step`` 个 TFR 点
        做均值压缩，得到 time_step 分辨率的时间特征。默认为0.1。
        None 表示不压缩，直接使用 TFR 原始时间分辨率。

        例：tfr_sfreq=300, time_step=0.1, time_range=(-1.5, 1.5)
            → 每帧时间窗 900 个 TFR 点，压缩为 30 个特征点。
    amp_feature : bool
        若提供，则在每帧"0 时刻"从原始 Raw 信号中采一个幅值样本，
        拼接在 TFR 特征的频率轴最前面（freq index 0），单位与 Raw 一致。
        Raw 的采样率可与 TFR 不同，会自动匹配最近样本。
        None 表示不包含幅值特征。
    if_show : bool
        若 True，调用 ``show_features(X)`` 可视化。

    Returns
    -------
    features : ``mne.time_frequency.EpochsTFRArray``
        ``features.data`` shape:
        ``(n_frames, n_channels, n_freqs_out, n_times_out)``。

    """
    # ------------------------------------------------------------------ #
    # 1. 解析标注中的 sample_start / sample_end
    # ------------------------------------------------------------------ #
    session = tfr.info['description']
    onsets = import_onsets(session=session)
    t_start = onsets["sample_start"]
    t_end   = onsets["sample_end"]
    fps = import_fps(session=session, stage='quan')

    # ------------------------------------------------------------------ #
    # 2. TFR 时间轴与基本参数
    # ------------------------------------------------------------------ #
    tfr_times  = tfr.times                         # (n_tfr_times,)
    tfr_sfreq  = tfr.info["sfreq"]                 # e.g. 300.0
    n_tfr_times = len(tfr_times)

    # ------------------------------------------------------------------ #
    # 4. 帧"0 时刻"索引（fps 决定步进，与 tfr_sfreq 无关）
    # ------------------------------------------------------------------ #
    idx_start = int(np.argmin(np.abs(tfr_times - t_start)))
    idx_end   = int(np.argmin(np.abs(tfr_times - t_end)))

    # 帧步进：在 TFR 时间轴上每隔 tfr_sfreq/fps 个点取一帧
    frame_step_tfr = tfr_sfreq / fps              # 可以是非整数
    frame_zero_indices = np.round(
        np.arange(idx_start, idx_end + 1, frame_step_tfr)
    ).astype(int)                                  # shape: (n_frames,)
    # 严格限制在 [idx_start, idx_end]（浮点 arange 可能多一个点）
    frame_zero_indices = frame_zero_indices[frame_zero_indices <= idx_end]

    n_frames = len(frame_zero_indices)

    # ------------------------------------------------------------------ #
    # 5. 时间窗口内偏移量：TFR 原始分辨率上的整点偏移
    # ------------------------------------------------------------------ #
    t0, t1 = time_range

    # ---- 5a. 确定压缩参数 ----
    if time_step is None or np.isclose(1 / time_step, tfr_sfreq):
        # 不压缩：直接用 TFR 分辨率
        bin_size = 1
        n_bins   = int(round((t1 - t0) * tfr_sfreq))  # 时间窗内 TFR 点数
    else:
        # 压缩：bin_size 个 TFR 点 → 1 个特征点
        bin_size = tfr_sfreq * time_step
        if not np.isclose(bin_size, round(bin_size)):
            raise ValueError(
                f"tfr_sfreq ({tfr_sfreq}) must be an integer multiple of "
                f"feature_sfreq ({1 /time_step}). Got ratio {bin_size:.4f}."
            )
        bin_size = int(round(bin_size))
        n_bins   = int(round((t1 - t0) / time_step))  # 压缩后特征点数
        n_tfr_window = n_bins * bin_size                   # 时间窗内 TFR 点数

    # 时间窗口起始偏移（TFR 点数，相对于帧"0 时刻"）
    off_start_tfr = int(round(t0 * tfr_sfreq))   # 负值 = 往过去偏移

    # 时间窗口内所有 TFR 点的偏移量，shape: (n_tfr_window,)
    if bin_size == 1:
        offsets_tfr = np.arange(off_start_tfr, off_start_tfr + n_bins)
    else:
        offsets_tfr = np.arange(off_start_tfr, off_start_tfr + n_tfr_window)

    # ------------------------------------------------------------------ #
    # 6. 取 TFR 数据、频率筛选、转 power_dB
    #    tfr_data shape: (n_channels, n_freqs_sel, n_tfr_times)
    # ------------------------------------------------------------------ #
    power_db = 10.0 * np.log10(np.maximum(tfr.data, 1e-30))

    if apply_baseline:
        tfr._data = power_db
        df_inv = import_involvement(session)
        tfr_rest = []
        for _, row in df_inv.query('Annotation == "motionless"').iterrows():
            tfr_r = tfr.copy().crop(tmin=row['Onset'] + t_start, tmax=row['Onset'] + row['Duration'] + t_start)
            tfr_rest.append(tfr_r)

        # 1. 把所有 rest 片段沿时间轴拼接，得到用于估计基线统计量的数据
        #    tfr_r.data 形状一般是 (n_channels, n_freqs, n_times)（AverageTFR）
        #    或 (n_epochs, n_channels, n_freqs, n_times)（EpochsTFR），时间轴始终是最后一维
        rest_data = np.concatenate([tfr_r.data for tfr_r in tfr_rest], axis=-1)

        # 2. 对每个 channel × freq，沿时间轴计算均值和标准差（keepdims 方便广播）
        rest_mean = rest_data.mean(axis=-1, keepdims=True)
        rest_std = rest_data.std(axis=-1, keepdims=True)

        # 3. 对完整的 tfr 做 z-score（等价于 apply_baseline 里 mode='zscore' 的公式）
        power_db = (power_db - rest_mean) / rest_std    

    n_channels, n_freqs_sel, _ = power_db.shape

    # ------------------------------------------------------------------ #
    # 7. 先 bin 压缩，再按帧索引取值（节省内存）
    #
    # 核心思路：
    #   帧步进 frame_step_tfr = tfr_sfreq / fps，时间窗口内每个 bin 的
    #   起始点相对于 TFR 时间轴的相位（模 bin_size 的余数）随帧变化。
    #   不同帧可能来自不同相位组，相位种数 = n_phases =
    #       bin_size / gcd(frame_step_tfr, bin_size)  （当 step 为整数时）
    #
    #   对 bin_size == 1（无压缩）退化为直接广播索引取值。
    #
    #   对 bin_size > 1：
    #     a) 枚举所有出现的相位 p（即 bin 起点 mod bin_size 的所有取值）
    #     b) 对每个相位 p，将 power_db 在时间轴上截取从 p 开始、步长1、
    #        长度为 bin_size 整数倍的子序列，reshape 后沿 bin_size 轴求均值，
    #        得到该相位的压缩 TFR，shape (n_ch, n_f, n_compressed_p)
    #     c) 对每帧按其相位查表取出对应压缩 TFR 上的 n_bins 个点
    #
    #   内存峰值从 O(n_ch × n_f × n_frames × n_tfr_window)
    #   降至     O(n_ch × n_f × n_tfr_times)  +  O(n_frames × n_bins)
    # ------------------------------------------------------------------ #
    if bin_size == 1:
        # ---- 无压缩：直接广播索引 ----
        time_indices = frame_zero_indices[:, None] + offsets_tfr[None, :]
        # 越界检测
        out_of_bounds = (time_indices < 0) | (time_indices >= n_tfr_times)
        n_oob = int(out_of_bounds.sum())
        if n_oob > 0:
            n_frames_oob = int(out_of_bounds.any(axis=1).sum())
            warnings.warn(
                f"时间索引越界：{n_oob} 个采样点（涉及 {n_frames_oob} 帧）超出 TFR 范围 "
                f"[0, {n_tfr_times})，对应位置将填充 NaN。\n"
                f"  · 起始端越界：{int((time_indices < 0).sum())} 个点"
                f"（sample_start 前不足 {-t0:.2f} 秒）\n"
                f"  · 末尾端越界：{int((time_indices >= n_tfr_times).sum())} 个点"
                f"（sample_end 后不足 {t1:.2f} 秒）",
                stacklevel=2,
            )
        safe_indices = np.clip(time_indices, 0, n_tfr_times - 1)
        # → (n_ch, n_f, n_frames, n_bins)
        X = power_db[:, :, safe_indices]
        if n_oob > 0:
            X = np.where(out_of_bounds[None, None, :, :], np.nan, X)

    else:
        # ---- 先 bin 压缩，再取帧 ----

        # 7a. 计算每帧第 0 个 bin 起点的相位（mod bin_size）
        #     bin_0_start[i] = frame_zero_indices[i] + off_start_tfr
        bin_0_starts = frame_zero_indices + off_start_tfr   # (n_frames,)，可含负值
        phases_per_frame = bin_0_starts % bin_size           # (n_frames,) ∈ [0, bin_size)
        unique_phases = np.unique(phases_per_frame)          # 通常远小于 bin_size

        # 7b. 为每个唯一相位预先计算压缩 TFR
        #     phase_to_compressed[p] = ndarray (n_ch, n_f, n_compressed_p)
        phase_to_compressed: dict[int, np.ndarray] = {}
        for p in unique_phases:
            p = int(p)
            # 从偏移 p 开始截取，保证长度是 bin_size 的整数倍
            raw_slice = power_db[:, :, p:]                  # (n_ch, n_f, L_p)
            L_p = raw_slice.shape[2]
            n_complete = L_p // bin_size
            if n_complete == 0:
                # 极端情况：相位太靠近末尾，直接存一个空数组
                phase_to_compressed[p] = raw_slice[:, :, :0]
                continue
            trimmed = raw_slice[:, :, : n_complete * bin_size]  # (n_ch, n_f, n_complete*bin_size)
            # reshape → mean: (n_ch, n_f, n_complete, bin_size) → (n_ch, n_f, n_complete)
            compressed = trimmed.reshape(
                n_channels, n_freqs_sel, n_complete, bin_size
            ).mean(axis=-1)
            phase_to_compressed[p] = compressed              # (n_ch, n_f, n_complete_p)

        # 7c. 按帧取值，组装 X
        #
        # 对每帧 i：
        #   相位 p = phases_per_frame[i]
        #   在压缩 TFR 中，第 j 个 bin 对应的索引：
        #     bin_start_tfr = bin_0_starts[i] + j * bin_size
        #     该 bin 在相位 p 的压缩序列里的索引 = (bin_start_tfr - p) / bin_size
        #                                       = (bin_0_starts[i] - p) / bin_size + j
        #     令 base_idx[i] = (bin_0_starts[i] - p) / bin_size，则
        #     第 j 个 bin 的压缩索引 = base_idx[i] + j
        #
        # 每帧需要取 n_bins 个连续压缩点，起始为 base_idx[i]。

        X_list = []
        oob_list = []

        for p in unique_phases:
            p = int(p)
            comp = phase_to_compressed[p]                   # (n_ch, n_f, n_complete_p)
            n_comp = comp.shape[2]

            frame_mask = phases_per_frame == p
            frames_p   = np.where(frame_mask)[0]            # 属于该相位的帧下标
            b0 = bin_0_starts[frames_p]                     # (n_frames_p,)

            # 压缩序列中的起始索引
            base_idx = (b0 - p) // bin_size                 # (n_frames_p,)，可含负数

            # 展开为 (n_frames_p, n_bins) 的索引矩阵
            comp_indices = base_idx[:, None] + np.arange(n_bins)[None, :]

            # 越界检测（压缩 TFR 坐标系下）
            oob = (comp_indices < 0) | (comp_indices >= n_comp)
            safe = np.clip(comp_indices, 0, n_comp - 1)

            # 取值：(n_ch, n_f, n_frames_p, n_bins)
            X_p = comp[:, :, safe]
            if oob.any():
                X_p = np.where(oob[None, None, :, :], np.nan, X_p)

            X_list.append((frames_p, X_p))
            oob_list.append(oob)

        # 汇总：按帧下标散布回完整数组
        X = np.empty((n_channels, n_freqs_sel, n_frames, n_bins), dtype=power_db.dtype)
        for frames_p, X_p in X_list:
            X[:, :, frames_p, :] = X_p

        # 越界警告（统一汇报）
        n_oob = sum(int(oob.sum()) for oob in oob_list)
        if n_oob > 0:
            n_frames_oob = sum(int(oob.any(axis=1).sum()) for oob in oob_list)
            # 重建原始 TFR 坐标系的越界信息用于报告
            _all_b0  = bin_0_starts
            _all_end = _all_b0 + n_tfr_window
            n_start_oob = int((_all_b0 < 0).sum() * n_bins)
            n_end_oob   = int((_all_end > n_tfr_times).sum() * n_bins)
            warnings.warn(
                f"时间索引越界：{n_oob} 个压缩 bin（涉及 {n_frames_oob} 帧）超出 TFR 范围 "
                f"[0, {n_tfr_times})，对应位置已填充 NaN。\n"
                f"  · 起始端越界：约 {n_start_oob} 个点"
                f"（sample_start 前不足 {-t0:.2f} 秒）\n"
                f"  · 末尾端越界：约 {n_end_oob} 个点"
                f"（sample_end 后不足 {t1:.2f} 秒）",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ #
    # 8. 调整维度顺序为 (n_frames, n_channels, n_freqs_sel, n_bins)
    # ------------------------------------------------------------------ #
    X = X.transpose(2, 0, 1, 3).copy()
    # (n_channels, n_freqs_sel, n_frames, n_bins) → (n_frames, n_channels, n_freqs_sel, n_bins)

    # ------------------------------------------------------------------ #
    # 9. 幅值特征（amp）：在帧"0 时刻"采 Raw 信号
    #     拼接在频率轴最前面，形状 (n_frames, n_channels, 1, n_bins)
    #     其中 n_bins 轴全部填充同一幅值（保持维度一致，展平后等价于原始标量特征）
    # ------------------------------------------------------------------ #
    if amp_feature:
        amp_raw = import_ecog_prep(session=session)
        raw_times  = amp_raw.times
        raw_sfreq  = amp_raw.info["sfreq"]
        raw_data   = amp_raw.get_data()          # (n_channels, n_raw_times)

        # 每帧"0 时刻"在原始信号上的绝对时间
        frame_abs_times = tfr_times[frame_zero_indices]

        # 最近样本索引（向下取整，对应 MATLAB 的 find(time < t, 1, 'last')）
        amp_indices = np.searchsorted(raw_times, frame_abs_times, side="right") - 1
        amp_indices = np.clip(amp_indices, 0, raw_data.shape[1] - 1)

        # amp: (n_channels, n_frames) → (n_frames, n_channels)
        amp = raw_data[:, amp_indices].T          # (n_frames, n_channels)

        # 扩展为 (n_frames, n_channels, 1, n_bins)：n_bins 轴广播复制
        amp_feat = np.broadcast_to(
            amp[:, :, None, None],
            (n_frames, n_channels, 1, n_bins if bin_size > 1 else len(offsets_tfr)),
        ).copy()
        # 拼接到频率轴最前面
        X = np.concatenate([amp_feat, X], axis=2)
        # → (n_frames, n_channels, 1 + n_freqs_sel, n_time_out)
    
    # ------------------------------------------------------------------ #
    # 10. 帧时间戳（相对于 sample_start）
    # ------------------------------------------------------------------ #
    frame_times = tfr_times[frame_zero_indices] - t_start
    metadata = pd.DataFrame({"frame_times": frame_times})
    # events = np.zeros((n_frames, 3))
    # events[:, 0] = tfr_times[frame_zero_indices] / time_step

    if if_show:
        show_features(X)

    # ------------------------------------------------------------------ #
    # 11. 构建 Features（EpochsTFRArray 子类）
    # ------------------------------------------------------------------ #
    # times 轴：特征时间窗内的相对时间（以帧"0 时刻"为原点）
    X = X.astype(np.float32)
    n_times_out = X.shape[3]
    feat_sfreq  = tfr_sfreq if bin_size == 1 else 1 / time_step
    feat_times  = t0 + np.arange(n_times_out) / feat_sfreq

    # freqs 轴：筛选后的频率（含 amp 时在最前面插入 0 Hz 占位）
    feat_freqs = tfr.freqs
    if amp_feature:
        feat_freqs = np.concatenate([[0.0], feat_freqs])

    # info：复用 tfr 的 info（已含正确的 ch_names / sfreq）
    feat_info = mne.create_info(
        ch_names=tfr.ch_names,
        sfreq=feat_sfreq,
        ch_types=['eeg'] * len(tfr.ch_names),
    )
    feat_info["description"] = tfr.info["description"]

    return mne.time_frequency.EpochsTFRArray(
        info=feat_info,
        data=X,
        times=feat_times,
        freqs=feat_freqs,
        method=getattr(tfr, "method", None),
        metadata=metadata,
    )

def show_features(X):
    # 检查特征是否有伪迹
    plt.figure(figsize=(10, 3))
    plt.plot(X.mean(axis=(1, 2)))
    plt.axis('off')
    plt.show()


def select_features(features):

    session = features.info["description"]
    fps = features.info['sfreq']
    tmin, tmax = features.times[[0, -1]]
    tmax += 1 / fps
    frame_times = features.metadata['frame_times']

    df_bad = import_bad(session)
    selection = select_epoch(frame_times + tmin, frame_times + tmax, 
                             df_bad['Onset'], df_bad['Onset'] + df_bad['Duration'])
    
    return selection

def select_coors(session, joint, times, query=None, involvement=None):
    
    df_miss = import_miss(session, query=query)

    df_miss = df_miss[df_miss['Joint'] == joint]

    annot_onset = df_miss['Onset'].to_numpy(dtype=float)
    annot_end = annot_onset + df_miss['Duration'].to_numpy(dtype=float)
    selection = select_epoch(times, times, annot_onset, annot_end)

    if involvement is not None:
        df_invol = import_involvement(session, query=f'Annotation == "{involvement}"')
        annot_onset = df_invol['Onset'].to_numpy(dtype=float)
        annot_end = annot_onset + df_invol['Duration'].to_numpy(dtype=float) 
        selection2 = select_epoch(times, times, annot_onset, annot_end, remove_annot=False) 
        selection = np.intersect1d(selection, selection2)      
    return selection


def select_features_and_coors(features, df_vel0, feature_selection=None, coor_selection=None, frame_range=(None, None)):
    if feature_selection is None:
        feature_selection = np.arange(len(features))
    if coor_selection is None:
        coor_selection = np.arange(df_vel0.shape[0])

    selection = np.intersect1d(feature_selection, coor_selection)
    if frame_range[0] is not None and frame_range[1] is not None:
        selection = selection[(selection >= frame_range[0]) & (selection <= frame_range[1])]

    features_sel = features.copy()[selection]
    df_vel_sel = df_vel0.copy().iloc[selection]
    return features_sel, df_vel_sel


def shuffle_X(X, shuffle_dims=(0, 1, 2, 3), seed=None, save_as=None):
    """
    X: (frame, channel, freq, time) 或任意维度数组，dim0 为样本维
    shuffle_dims: 0 表示打乱 frame 顺序（全局 permutation）
                  1/2/3 表示对应特征维在每个样本内独立打乱
    save_as: 若给出路径，把复现所需的索引存为 pickle
    """
    rng = np.random.default_rng(seed)
    X = X.copy()
    index_dict = {}

    # dim 0: frame 顺序整体打乱
    if 0 in shuffle_dims:
        N = X.shape[0]
        perm0 = rng.permutation(N)
        X = X[perm0]
        index_dict[0] = perm0

    # dim 1,2,3: 每个 frame 独立打乱该维内部顺序
    for d in range(1, X.ndim):
        if d in shuffle_dims:
            X, perm_idx = shuffle(X, d, rng)
            index_dict[d] = perm_idx

    if save_as is not None:
        with open(save_as, "wb") as f:
            pickle.dump(index_dict, f)

    return X


def apply_shuffle(X, path):
    """
    用 load 回来的 index_dict 复现打乱后的 X。
    注意：dim0 的 permutation 必须最先应用（和 shuffle_X 里的顺序一致），
    之后各特征维按 index_dict 里保存的每样本索引重排。
    """

    with open(path, "rb") as f:
        index_dict = pickle.load(f)
    
    X = X.copy()

    if 0 in index_dict:
        X = X[index_dict[0]]

    for d, perm_idx in index_dict.items():
        if d == 0:
            continue
        X = np.moveaxis(X, d, 1)
        N, size_d = X.shape[0], X.shape[1]
        reshaped = perm_idx.reshape(N, size_d, *([1] * (X.ndim - 2)))
        X = np.take_along_axis(X, reshaped, axis=1)
        X = np.moveaxis(X, 1, d)

    return X