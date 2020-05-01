import numpy as np
import mne


def apply_reg_artifact_remover(raw):
    """EOG artifact removal based on 'A fully automated correction method of EOG artifacts in EEG recordings'."""
    raw_data = raw.get_data()

    autocov_raw = mne.compute_raw_covariance(raw)
    autocov_raw_matrix = autocov_raw.data

    eog_autocov = autocov_raw_matrix[3:6, 3:6]
    eog_eeg_crosscov = autocov_raw_matrix[0:3, 3:6]

    b = np.matmul(np.linalg.inv(eog_autocov), eog_eeg_crosscov).T
    raw_eeg = raw_data[:3, :] - np.matmul(b, raw_data[:3, :])
    raw[:3, :] = raw_eeg

    return raw
