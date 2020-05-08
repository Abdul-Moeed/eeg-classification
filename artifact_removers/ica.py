# import mne
from mne.preprocessing import ICA, create_eog_epochs
import mne.channels as chan

def ica_preprocess(raw, vis, lp_threshold):
    """add montage and filter slow drifts."""
    data_path = __file__ + '/../../data/'
    raw.set_channel_types({'EOG:ch01': 'eog', 'EOG:ch02': 'eog', 'EOG:ch03': 'eog'})

    # Read montage/digitisation points
    raw_fname = data_path + 'GrazIV2B_montage.elc'
    montage = chan.read_custom_montage(raw_fname)
    raw.set_montage(montage)

    # Get a summary of how the ocular artifact manifests across each channel type
    eog_evoked = create_eog_epochs(raw).average()
    eog_evoked.apply_baseline(baseline=(None, -0.2))

    if vis:
        eog_evoked.plot_joint()

    # Filtering to remove slow drifts
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=lp_threshold, h_freq=None)

    return filt_raw, eog_evoked


def remove_components(ica, raw, eog_evoked, cor_threshold, vis):
    """Remove component that correlates with EOG and reconstruct signal without it."""

    # Find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=cor_threshold)
    ica.exclude = eog_indices

    if vis:
        # Barplot of IC component "EOG match" scores
        ica.plot_scores(eog_scores)

        # Plot diagnostics
        ica.plot_properties(raw, picks=eog_indices)

        # Plot ICs applied to raw data, with EOG matches highlighted
        ica.plot_sources(raw)

        # Plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
        ica.plot_sources(eog_evoked)

    # Reconstruct signal
    reconst_raw = raw.copy()
    ica.apply(reconst_raw)

    if vis:
        raw.plot()
        reconst_raw.plot()

    return reconst_raw


def apply_ica(raw, lp_threshold, cor_threshold, vis=False):
    filt_raw, eog_evoked = ica_preprocess(raw, vis, lp_threshold)

    # Fitting and plotting the ICA solution
    ica = ICA(n_components=3, random_state=97)
    ica.fit(filt_raw)

    if vis:
        raw.load_data()
        ica.plot_sources(raw)
        ica.plot_components()

    return remove_components(ica, raw, eog_evoked, cor_threshold, vis)
