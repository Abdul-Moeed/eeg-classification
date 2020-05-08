# Classifier demo
import mne
from mne import io
import numpy as np

import os


# jupyter notebook --notebook-dir="D:/Box/Programs/PythonProjects/eeg-classification"

# from estimators.linear import train_linear as estimator
from estimators.svm import train_svm as estimator
from artifact_removers.ica import apply_ica as artifact_remover
# from artifact_removers.regression_based import apply_reg_artifact_remover as artifact_remover

import pickle

print(__doc__)


def epoch_data(data_path):
    # Set params
    raw_fname = data_path
    tmin, tmax = 2, 4  # time chosen from jupyter notebook
    event_id = {'left': 10, 'right': 11}

    # Setup for reading the raw data
    raw = io.read_raw_gdf(raw_fname, preload=True)

    # Remove artifacts
    raw = artifact_remover(raw, lp_threshold=6, cor_threshold=1.2, vis=False)

    raw.filter(4, 12, fir_design='firwin')  # extract alpha band (see jupyter notebook)
    events, _ = mne.events_from_annotations(raw)

    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax - 1 / raw.info['sfreq'], proj=True, baseline=None,
                        preload=True, picks=[0, 2])  # use only C3 and C4, they are different
    # MNE slicing for epochs INCLUDES the upper limit!!!
    labels = epochs.events[:, -1]

    return epochs, labels


def train(data_path):
    # epoch data
    epochs, labels = epoch_data(data_path)

    # fit classifier
    best_est = estimator(epochs, labels)

    return best_est


def test(data_path):
    # epoch data
    epochs, labels = epoch_data(data_path)
    X = epochs.get_data()
    X_sample = X[:40, :, :]
    y_sample = labels[:40]
    print(X.shape, labels.shape, X_sample.shape, y_sample.shape)

    # test classifier of subject 1
    clf_1 = pickle.load(open('models/trained_svm_clf_B0101T.gdf.sav', 'rb'))
    print("Predict classifier 1: ", clf_1.score(X_sample, y_sample))

    # test classifier of subject 2
    clf_2 = pickle.load(open('models/trained_svm_clf_B0202T.gdf.sav', 'rb'))
    print("Predict classifier 2: ", clf_2.score(X_sample, y_sample))


def main():
    # Get data path
    data_file = 'B0101T.gdf'
    data_path = os.getcwd() + '/data/' + data_file

    # train model
    model = train(data_path)

    # save model
    with open('models/trained_svm_clf_' + data_file + '.sav', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()


import gumpy
a = gumpy.classify('LogisticRegression', 0, 0, 0, 0)
gumpy.classification.vote(0, 0, 0, 0, 'soft', False, )
