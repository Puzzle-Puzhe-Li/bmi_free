def bilateral_reref(raw):
    raw1 = raw.copy().pick_channels([ch for ch in raw.ch_names if 'L' in ch])
    raw1.set_eeg_reference('average')
    raw2 = raw.copy().pick_channels([ch for ch in raw.ch_names if 'R' in ch])
    raw2.set_eeg_reference('average')

    raw = raw1.add_channels([raw2])
    return raw