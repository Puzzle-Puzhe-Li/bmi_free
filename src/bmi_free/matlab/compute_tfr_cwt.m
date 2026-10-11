function tfr = compute_tfr_cwt(data, fs, freq_bin_edge, time_in, time_out, voices)
% COMPUTE_TFR_CWT  Compute time-frequency representation via Morse CWT.
%
% Inputs
% ------
%   data          : [n_samples x n_channels]  ECoG signal (already time-delay corrected)
%   fs            : scalar  sampling rate of `data` (Hz), e.g. 2000
%   freq_bin_edge : [1 x (n_freq+1)]  edges of frequency bins (Hz), e.g. 5:10:145
%   time_in       : [n_samples x 1]  time axis of `data` (s)
%   time_out      : [n_out x 1]      target time axis for the output (s)
%   voices        : (optional) voices-per-octave for Morse wavelet (default 32)
%
% Output
% ------
%   tfr : [n_out x n_channels x n_freq]  power (sum of |CWT| within each freq bin),
%         interpolated onto time_out
%
% Notes
% -----
%   - Frequency binning uses sum (not mean) of CWT magnitudes inside each bin,
%     consistent with the original extract_features.m.
%   - Temporal resampling uses oneD_gridded_interpolation (must be on MATLAB path)
%     or falls back to interp1 if that function is unavailable.

if nargin < 6 || isempty(voices)
    voices = 32;
end

n_channels        = size(data, 2);
n_freq            = length(freq_bin_edge) - 1;
fmin              = freq_bin_edge(1);
fmax              = freq_bin_edge(end);

scalos = cell(1, n_channels);

for ch = 1:n_channels
    % ---- CWT (Morse wavelet) ------------------------------------------------
    [wt, f] = cwt(data(:, ch), 'morse', fs, ...
                  'VoicesPerOctave', voices, ...
                  'FrequencyLimits', [fmin, fmax]);
    wt = abs(wt);   % magnitude  [n_cwt_freqs x n_samples]

    % ---- Frequency binning --------------------------------------------------
    bin = discretize(f, freq_bin_edge);

    binned_wt = zeros(size(wt, 2), n_freq);   % [n_samples x n_freq]
    for k = 1:n_freq
        idx = (bin == k);
        if any(idx)
            binned_wt(:, k) = sum(wt(idx, :), 1);
        end
    end

    % ---- Temporal interpolation onto target grid ----------------------------
    if exist('oneD_gridded_interpolation', 'file')
        [binned_wt, ~] = oneD_gridded_interpolation(binned_wt, time_in, time_out, 1);
    else
        % Fallback: linear interp1 column-by-column
        binned_wt_out = zeros(length(time_out), n_freq);
        for k = 1:n_freq
            binned_wt_out(:, k) = interp1(time_in, binned_wt(:, k), time_out, 'linear', 0);
        end
        binned_wt = binned_wt_out;
    end

    scalos{ch} = binned_wt;   % [n_out x n_freq]
end

% ---- Stack channels ---------------------------------------------------------
% cat along dim-3 → [n_out x n_freq x n_channels], then permute to [n_out x n_channels x n_freq]
tfr = permute(cat(3, scalos{:}), [1, 3, 2]);

end
