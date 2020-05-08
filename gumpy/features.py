

from .classification import available_classifiers
import matplotlib.pyplot as plt
import sklearn.decomposition
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import numpy as np


def sequential_feature_selector(features, labels, classifier, k_features, kfold, selection_type, plot=True, **kwargs):
    """Sequential feature selection to reduce the number of features.

    The function reduces a d-dimensional feature space to a k-dimensional
    feature space by sequential feature selection. The features are selected
    using ``mlxtend.feature_selection.SequentialFeatureSelection`` which
    essentially selects or removes a feature from the d-dimensional input space
    until the preferred size is reached.

    The function will pass ``ftype='feature'`` and forward ``features`` on to a
    classifier's ``static_opts`` method.

    Args:
        features: The original d-dimensional feature space
        labels: corresponding labels
        classifier (str or object): The classifier which should be used for
            feature selection. This can be either a string (name of a classifier
            known to gumpy) or an instance of a classifier which adheres
            to the sklearn classifier interface.
        k_features (int): Number of features to select
        kfold (int): k-fold cross validation
        selection_type (str): One of ``SFS`` (Sequential Forward Selection),
            ``SBS`` (Sequential Backward Selection), ``SFFS`` (Sequential Forward
            Floating Selection), ``SBFS`` (Sequential Backward Floating Selection)
        plot (bool): Plot the results of the dimensinality reduction
        **kwargs: Additional keyword arguments that will be passed to the
            Classifier instantiation

    Returns:
        A 3-element tuple containing

        - **feature index**: Index of features in the remaining set
        - **cv_scores**: cross validation scores during classification
        - **algorithm**: Algorithm that was used for search

    """

    # retrieve the appropriate classifier
    if isinstance(classifier, str):
        if not (classifier in available_classifiers):
            raise ClassifierError("Unknown classifier {c}".format(c=classifier.__repr__()))

        kwopts = kwargs.pop('opts', dict())
        # opts = dict()

        # retrieve the options that we need to forward to the classifier
        # TODO: should we forward all arguments to sequential_feature_selector ?
        opts = available_classifiers[classifier].static_opts('sequential_feature_selector', features=features)
        opts.update(kwopts)

        # XXX: now merged into the static_opts invocation. TODO: test
        # if classifier == 'SVM':
        #     opts['cross_validation'] = kwopts.pop('cross_validation', False)
        # elif classifier == 'RandomForest':
        #     opts['cross_validation'] = kwopts.pop('cross_validation', False)
        # elif classifier == 'MLP':
        #     # TODO: check if the dimensions are correct here
        #     opts['hidden_layer_sizes'] = (features.shape[1], features.shape[2])
        # get all additional entries for the options
        # opts.update(kwopts)

        # retrieve a classifier object
        classifier_obj = available_classifiers[classifier](**opts)

        # extract the backend classifier
        clf = classifier_obj.clf
    else:
        # if we received a classifier object we'll just use this one
        clf = classifier.clf


    if selection_type == 'SFS':
        algorithm = "Sequential Forward Selection (SFS)"
        sfs = SFS(clf, k_features, forward=True, floating=False,
                verbose=2, scoring='accuracy', cv=kfold, n_jobs=-1)

    elif selection_type == 'SBS':
        algorithm = "Sequential Backward Selection (SBS)"
        sfs = SFS(clf, k_features, forward=False, floating=False,
                verbose=2, scoring='accuracy', cv=kfold, n_jobs=-1)

    elif selection_type == 'SFFS':
        algorithm = "Sequential Forward Floating Selection (SFFS)"
        sfs = SFS(clf, k_features, forward=True, floating=True,
                verbose=2, scoring='accuracy', cv=kfold, n_jobs=-1)

    elif selection_type == 'SBFS':
        algorithm = "Sequential Backward Floating Selection (SFFS)"
        sfs = SFS(clf, k_features, forward=True, floating=True,
                verbose=2, scoring='accuracy', cv=kfold, n_jobs=-1)

    else:
        raise Exception("Unknown selection type '{}'".format(selection_type))


    pipe = make_pipeline(StandardScaler(), sfs)
    pipe.fit(features, labels)
    subsets = sfs.subsets_
    feature_idx = sfs.k_feature_idx_
    cv_scores = sfs.k_score_

    if plot:
        fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
        plt.ylim([0.5, 1])
        plt.title(algorithm)
        plt.grid()
        plt.show()

    return feature_idx, cv_scores, algorithm, sfs, clf

#Zero crossings: it is the number of times the waveform crosses zero. This feature provides an approximate estimation of frequency domain properties. The threshold avoids counting zero crossings induced by noise. 

#MAV
def mav(segment):
    mav = np.mean(np.abs(segment))
    return mav
#rms
def RMS(segment):
    rms = np.sqrt(np.mean(np.power(segment,2)))
    return rms

#var
def var(segment):
    var = np.var(segment)
    return var

#Simple square integral: it gives a measure of the energy of the EMG signal. 
def ssi(segment):
    ssi = np.sum(np.abs(np.power(segment,2)))
    return ssi

#Waveform length: it is the cumulative length of the waveform over the segment. This feature is related to the signal amplitude, frequency and time. 

def wl(segment):
    wl = np.sum(np.abs(np.diff(segment)))
    return wl

zc = 0

def zcx(segment):
    global zc
    nz_segment = []
    nz_indices = np.nonzero(segment)[0] # Finds the indices of the segment with nonzero values
    for i in nz_indices:
        nz_segment.append(segment[i]) # The new segment contains only nonzero values    
    N = len(nz_segment)
    
    for n in range(N-1):
        if((nz_segment[n]*nz_segment[n+1]<0) and np.abs(nz_segment[n]-nz_segment[n+1]) >= 0.001):
            zc = zc + 1
    return zc

#Slope sign changes: it is similar to the zero crossings feature. It also provides information about the frequency content of the signal. It is calculated as follows
ssc = 0

def sscx(segment):
    N = len(segment)
    global ssc
    for n in range(1,N-1):
        if (segment[n]-segment[n-1])*(segment[n]-segment[n+1])>=0.001:
            ssc = ssc + 1
    return ssc

#Willison amplitude: it is the number of times that the difference of the amplitude between to adjacent data points exceed a predefined threshold. This feature provides information about the muscle contraction level.

wamp = 0

def wampx(segment):
    N = len(segment)
    global wamp
    for n in range(N-1):
        if np.abs(segment[n]-segment[n+1])>=50:
            wamp = wamp + 1
    return wamp

def features_calculation(signal, fs, window_size, window_shift):
    """Root Mean Square.

    Args:
        signal (array_like): TODO
        fs (int): Sampling frequency
        window_size: TODO
        window_shift: TODO

    Returns:
        TODO:
    """
    duration = len(signal)/fs
    n_features = int(duration/(window_size-window_shift))

    rms_features = np.zeros(n_features)
    mav_features = np.zeros(n_features)
    ssi_features = np.zeros(n_features)
    var_features = np.zeros(n_features)
    zc_features = np.zeros(n_features)
    wl_features = np.zeros(n_features)
    ssc_features = np.zeros(n_features)
    wamp_features = np.zeros(n_features)
    
    global ssc
    global zc
    global wamp

    for i in range(n_features):
        idx1 = int((i*(window_size-window_shift))*fs)
        idx2 = int(((i+1)*window_size-i*window_shift)*fs)
        rms = np.sqrt(np.mean(np.square(signal[idx1:idx2])))
        mav = np.mean(np.abs(signal[idx1:idx2]))
        var = np.var(signal[idx1:idx2])
        ssi = np.sum(np.abs(np.square(signal[idx1:idx2])))
        ssc = sscx(signal[idx1:idx2])
        #Waveform length: it is the cumulative length of the waveform over the segment. This feature is related to the signal amplitude, frequency and time. 
        wl = np.sum(np.abs(np.diff(signal[idx1:idx2])))
        zc = zcx(signal[idx1:idx2])
        wamp = wampx(signal[idx1:idx2])
        wamp_features[i] = wamp
        wl_features[i] = wl
        zc_features[i] = zc             
        ssi_features[i] = ssi
        ssc_features= ssc
        mav_features[i] = mav
        var_features[i] = var
        rms_features[i] = rms

    return wamp_features, wl_features, zc_features, ssi_features, mav_features, var_features, rms_features   



def features_extraction(data, trialList, window_size, window_shift):
    if window_shift > window_size:
        raise ValueError("window_shift > window_size")

    fs = data.sampling_freq
    
    n_features = int(data.duration/(window_size-window_shift))
    
    X = np.zeros((len(trialList), n_features*4))
    
    t = 0
    for trial in trialList:
        # x3 is the worst of all with 43.3% average performance
        x1=features_calculation(trial[0], fs, window_size, window_shift)
        x2=features_calculation(trial[1], fs, window_size, window_shift)
        x3=features_calculation(trial[2], fs, window_size, window_shift)
        x4=features_calculation(trial[3], fs, window_size, window_shift)
        x=np.concatenate((x1, x2, x3, x4))
        X[t, :] = np.array([x])
        t += 1
    return X


def dwt_features(data, trials, level, sampling_freq, w, n, wavelet):
    """Extract discrete wavelet features
    Args:
        data: 2D (time points, Channels)
        trials: Trials vector
        lLevel: level of DWT decomposition
        sampling_freq: Sampling frequency
    Returns:
        The features matrix (Nbre trials, Nbre features)
    """

    # number of features per trial
    n_features = 9
    # allocate memory to store the features
    X = np.zeros((len(trials), n_features))

    # Extract Features
    for t, trial in enumerate(trials):
        signals = data[trial + fs*4 + (w[0]) : trial + fs*4 + (w[1])]
        coeffs_c3 = pywt.wavedec(data = signals[:,0], wavelet=wavelet, level=level)
        coeffs_c4 = pywt.wavedec(data = signals[:,1], wavelet=wavelet, level=level)
        coeffs_cz = pywt.wavedec(data = signals[:,2], wavelet=wavelet, level=level)

        X[t, :] = np.array([
            np.std(coeffs_c3[n]), np.mean(coeffs_c3[n]**2),
            np.std(coeffs_c4[n]), np.mean(coeffs_c4[n]**2),
            np.std(coeffs_cz[n]), np.mean(coeffs_cz[n]**2),
            np.mean(coeffs_c3[n]),
            np.mean(coeffs_c4[n]),
            np.mean(coeffs_cz[n])])

    return X


def PCA_dim_red(features, var_desired):
    """Dimensionality reduction of features using PCA.

    Args:
        features (matrix (2d np.array)): The feature matrix
        var_desired (float): desired preserved variance

    Returns:
        features with reduced dimensions

    """
    # PCA
    pca = sklearn.decomposition.PCA(n_components=features.shape[1]-1)
    pca.fit(features)
    # print('pca.explained_variance_ratio_:\n',pca.explained_variance_ratio_)
    var_sum = pca.explained_variance_ratio_.sum()
    var = 0
    for n, v in enumerate(pca.explained_variance_ratio_):
        var += v
        if var / var_sum >= var_desired:
            features_reduced = sklearn.decomposition.PCA(n_components=n+1).fit_transform(features)
            return features_reduced

      
    


