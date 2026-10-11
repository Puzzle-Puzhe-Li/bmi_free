def bilateral_reref(raw):
    """
    双侧平均重参考：对左半球 (含 'L') 和右半球 (含 'R') 通道分别做平均参考。

    Parameters
    ----------
    raw : mne.io.Raw
        原始 ECoG 数据，通道名需包含 'L' 或 'R' 以区分左右半球。

    Returns
    -------
    mne.io.Raw
        重参考后的数据，左半球和右半球分别以各自平均为参考。
    """
    raw1 = raw.copy().pick_channels([ch for ch in raw.ch_names if 'L' in ch])
    raw1.set_eeg_reference('average')
    raw2 = raw.copy().pick_channels([ch for ch in raw.ch_names if 'R' in ch])
    raw2.set_eeg_reference('average')

    raw = raw1.add_channels([raw2])
    return raw