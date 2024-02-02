
import numpy as np
import pandas as pd

from Code1.project_configuration import get_parameter

from scipy.signal import welch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def get_average_amplitude(signal, samplerate):

    subsample = get_parameter('subsample_interval')
    n_ma_sample = round(samplerate * get_parameter('moving_average_win_duration') / subsample)

    # Make a copy that we can modify without messing up the original data
    # Take absolute value and a moving average to find the average amplitude along parts of the signal
    signal_pp = signal[0::subsample].copy()
    signal_pp = np.abs(signal_pp);
    signal_pp = np.convolve(signal_pp, np.ones(n_ma_sample), 'same') / n_ma_sample

    return signal_pp

def estimate_kmeans_model(data, n_clusters):

    # Initialize the centroids so that the clusters corresponds to RPMs (0, 4000, 10000) in that order
    if n_clusters == 2:
        init_centroids = np.array([0, 0.5])*data.max()
    elif n_clusters == 3:
        init_centroids = np.array([0, 0.05, 0.7])*data.max()

    kmeans = KMeans(n_clusters=n_clusters, n_init=1, init=init_centroids[:, np.newaxis])
    kmeans.fit(data[:, np.newaxis])

    return kmeans

def remove_too_short_segments(labels, samplerate):

    labels_new = labels.copy()

    subsample = get_parameter('subsample_interval')
    n_sample_lim = round(samplerate * get_parameter('min_segment_duration')) / subsample

    # Detect short segments
    labels_new[[0, -1]] = 0
    changes = np.abs(np.sign(np.diff(np.hstack([0, labels_new, 0]))))
    change_indices = np.where(changes)[0]
    durations = np.diff(change_indices)
    too_short_segments = np.where(durations<n_sample_lim)[0]

    # Re-assign all short segments as belonging to the first class
    for change_idx in too_short_segments:
        start_idx = change_indices[change_idx]
        stop_idx = change_indices[change_idx+1]
        labels_new[start_idx:stop_idx] = 0

    return labels_new

def extract_onset_and_offset_indices_long(labels, samplerate, machine_id):

    subsample = get_parameter('subsample_interval')
    n_exclude = round(samplerate * get_parameter('ramp_up_and_down_time')) / subsample

    # Find changes
    changes = np.abs(np.sign(np.diff(np.hstack([0, labels]))))
    change_indices = np.where(changes)[0];
    labels_after_change = labels[change_indices]

    # Exctract onset and offset data
    occurence_data = []
    n_per_rpm = np.array([1, 1, 1])
    label_rpm_map = [0, 4000, 10000]
    for pos in np.where(labels_after_change>0)[0]:
        label_tmp = labels_after_change[pos]
        occurence_data.append((machine_id,
                               label_rpm_map[label_tmp],
                               n_per_rpm[label_tmp],
                               round(subsample*(change_indices[pos]+n_exclude)),
                               round(subsample*(change_indices[pos+1]-1-n_exclude))))
        n_per_rpm[label_tmp] += 1

    # Convert to a pandas data frame and store the table to as csv file.
    df = pd.DataFrame(occurence_data, columns =['Machine_ID', 'RPM', 'Trial idx', 'Onset', 'Offset'])

    return df

def dense_dft_with_serch(signal, samplerate, df_dense, freq_interval):

    # Set up n and k values for the DFT matrix
    freq_dense = np.arange(0, samplerate, df_dense)
    N = np.size(signal)
    N_dense = freq_dense.size
    n_vals = np.arange(N)
    k_vals_dense = np.arange(N_dense)

    # Prune k values outside of the wanted frequency range
    k_vals_dense = k_vals_dense[(freq_dense > freq_interval[0]) & (freq_dense < freq_interval[1])]
    freq_dense = freq_dense[(freq_dense > freq_interval[0]) & (freq_dense < freq_interval[1])]

    # Initialize the end points for the divide and conquer search
    low_idx = 0
    high_idx = k_vals_dense.size - 1
    delta_idx = high_idx - low_idx
    n_vals_tmp = 2*np.pi/N_dense*n_vals
    nk_vec_low = k_vals_dense[low_idx]*n_vals_tmp
    nk_vec_high = k_vals_dense[high_idx]*n_vals_tmp
    dft_low = np.abs( 1./N * (np.cos(nk_vec_low)@signal + np.sin(nk_vec_low)@signal*1j)  )
    dft_high = np.abs( 1./N * (np.cos(nk_vec_high)@signal + np.sin(nk_vec_high)@signal*1j)  )

    # Initial the dft array
    dft_dense = np.zeros(k_vals_dense.size)
    dft_dense[low_idx] = dft_low
    dft_dense[high_idx] = dft_high

    # Divide and conqur search to find the frequency with the highest inner product with the signal
    slit_ratio = 3
    converged = False
    while not converged:

        if delta_idx < 10:
            slit_ratio = 2

        if dft_low > dft_high:
            high_idx -= int(np.ceil(delta_idx/slit_ratio))
            nk_vec_high = k_vals_dense[high_idx]*n_vals_tmp
            dft_high = np.abs( 1./N * (np.cos(nk_vec_high)@signal + np.sin(nk_vec_high)@signal*1j)  )
            dft_dense[high_idx] = dft_high
        else:
            low_idx += int(np.floor(delta_idx/slit_ratio))
            nk_vec_low = k_vals_dense[low_idx]*n_vals_tmp
            dft_low = np.abs( 1./N * (np.cos(nk_vec_low)@signal + np.sin(nk_vec_low)@signal*1j)  )
            dft_dense[low_idx] = dft_low

        delta_idx = high_idx - low_idx


        if delta_idx < 2:
            converged = True

    return dft_dense, freq_dense

def dense_dft(signal, samplerate, df_dense, freq_interval):

    # Set up n and k values for the DFT matrix
    freq_dense = np.arange(0, samplerate, df_dense)
    N = np.size(signal)
    N_dense = freq_dense.size
    n_vals = np.arange(N)
    k_vals_dense = np.arange(N_dense)

    # Prune k values outside of the wanted frequency range
    k_vals_dense = k_vals_dense[(freq_dense > freq_interval[0]) & (freq_dense < freq_interval[1])]
    freq_dense = freq_dense[(freq_dense > freq_interval[0]) & (freq_dense < freq_interval[1])]

    # Compute the dft (incredibly slow to compute the sin and cos functions)
    nk_mat_dense = 2*np.pi/N_dense*np.outer(k_vals_dense, n_vals)
    cos_mat_dense = np.cos(nk_mat_dense)
    sin_mat_dense = np.sin(nk_mat_dense)
    dft_dense = 1./(N) * (cos_mat_dense @ signal + sin_mat_dense @ signal * 1j)

    return dft_dense, freq_dense

def compute_spectra_and_real_rpm(signal, samplerate, onset, offset):

    n_min = get_parameter('min_real_rpm_samples')
    delta_freq = get_parameter('frequency_resolution')
    cutoff_freq = get_parameter('cutoff_frequency')
    perform_time_warp = get_parameter('time_warp')
    frequencies = np.arange(0, cutoff_freq, delta_freq)

    # FFT on the whole segment to find the peak
    data_segment = signal[onset:offset]
    welch_freq_pad, welch_pxx_pad = welch(data_segment, samplerate, scaling='spectrum', nperseg=data_segment.size)
    peak_idx = welch_pxx_pad[welch_freq_pad<=200].argmax()
    rough_rpm_estimate = welch_freq_pad[peak_idx] * 60
    intended_rpm = int( np.round(rough_rpm_estimate/1e3) * 1e3 )
    rpm_freq = intended_rpm / 60.

    # Normalize (this seems to work better than normalizing by rpm amplitude)
    data_segment = data_segment / np.max(np.abs(data_segment))

    # Use a dense DFT matrix over the selected frequency to get a more accurate estimate of the real RPM values
    # This might not be needed if the data segment is long enough to begin with
    #df_dense = 0.1
    #freq_interval_dense = [welch_freq_pad[peak_idx-1], welch_freq_pad[peak_idx+1]]
    #dft_dense, freq_dense = dense_dft_with_serch(data_segment, samplerate, df_dense, freq_interval_dense)
    #real_rpm = freq_dense[np.argmax(np.abs(dft_dense))] * 60
    real_rpm = rough_rpm_estimate

    # Use Welch's method for averaging periodograms to get the desired frequency resolution
    if perform_time_warp:
        time_warp = intended_rpm / real_rpm  # stretch the time axis to ensure that all segments have the same RPM
    else:
        time_warp = 1
    samplerate_new = samplerate*time_warp
    #samplerate_new = samplerate*1
    welch_freq_tmp, welch_pxx_tmp = welch(data_segment - data_segment.mean(), samplerate_new, scaling='spectrum', nperseg=round(samplerate_new/delta_freq))
    # Normalize to have unity for the dominating RPM frequency
    #welch_pxx_tmp = welch_pxx_tmp / welch_pxx_tmp.max()
    # Only keep data below the cutoff frequency
    spectra = welch_pxx_tmp[0:frequencies.size]

    return spectra, real_rpm

def plot_segment_labels_and_signal(signal, df, samplerate, filename=None):

    y_max = np.abs(signal).max()
    n_plot_samples = min([signal.shape[0], samplerate*500])
    time = np.arange(signal.size) / samplerate

    fig, ax = plt.subplots(1, 1);
    for i in range(df.shape[0]):
        # Highlight the segment onset and offset with RPM specific color
        if df.iloc[i]['RPM'] == 4000:
            color_tmp = np.array([1, 0, 0])
        elif df.iloc[i]['RPM'] == 10000:
            color_tmp = np.array([0, 0, 1])
        else:
            color_tmp = np.array([0, 1, 0])
        x_tmp = np.array([df.iloc[i]['Onset'], df.iloc[i]['Offset'], df.iloc[i]['Offset'], df.iloc[i]['Onset']], dtype=np.int64)
        ax.fill(time[x_tmp], y_max*np.array([-1, -1, 1, 1]), facecolor=color_tmp, edgecolor=None, alpha=0.5)

    # Plot a subsampled verison of the raw signal on top
    ax.plot(time[:n_plot_samples:100], signal[:n_plot_samples:100], 'k-')
    ax.set_xlabel('Time (s)')
    ax.set_xlim([0, time[n_plot_samples]]);
    ax.set_ylim([-y_max, y_max]);
    ax.set_yticklabels([])

    if filename:
        fig.savefig(filename + '.png', format='png')

def plot_average_spectra(df, filename=None):

    first_spectra_column = df.columns.get_loc(0.0)

    frequencies = df.columns[first_spectra_column:].to_numpy()
    all_spectras = df.iloc[:, first_spectra_column:].to_numpy()

    average_low_spectra = all_spectras[df['RPM'] == 4000, :].mean(axis=0)
    average_high_spectra = all_spectras[df['RPM'] == 10000, :].mean(axis=0)

    fig, ax = plt.subplots(2, 1);
    y_max = 1.1*average_low_spectra[100:].max()
    ax[0].plot(frequencies, average_low_spectra, 'r', label='RPM: 4000')
    ax[0].set_ylim([0, y_max])
    ax[0].set_yticklabels([])
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].legend()

    y_max = 1.1*average_high_spectra[100:].max()
    ax[1].plot(frequencies, average_high_spectra, 'b', label='RPM: 10000')
    ax[1].set_ylim([0, y_max])
    ax[1].set_yticklabels([])
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].legend()

    if filename:
        fig.savefig(filename + '.png', format='png')
