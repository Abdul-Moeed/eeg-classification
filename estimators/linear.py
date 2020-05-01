import numpy as np

from mne.decoding import Vectorizer
from mne.time_frequency import psd_multitaper
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, ShuffleSplit

# import a linear classifier from mne.decoding
from mne.decoding import LinearModel, PSDEstimator


def train_linear(epochs, labels):
    # Logistic Regression for EEG
    X = epochs.pick_types(meg=False, eeg=True)
    X_data = X.get_data()

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    # Define a unique pipeline to sequentially:
    clf = make_pipeline(
        PSDEstimator(epochs.info['sfreq'], fmin=6, fmax=14, n_jobs=1),
        Vectorizer(),  # 1) vectorize across frequency and channels
        StandardScaler(),  # 2) normalize features across trials
        LinearModel(
            LogisticRegression(solver='lbfgs')))  # 3) fits a logistic regression

    scores = cross_validate(clf, X_data, labels, cv=cv, n_jobs=1, return_estimator=True, return_train_score=False)

    print('Mean: {0}, Std: {1}'.format(scores['test_score'].mean(), scores['test_score'].std()))

    # best_est = scores['estimator'][scores['test_score'].argmax()]
    best_est = clf.fit(X_data, labels)  # refit the estimator on the whole set

    return best_est
