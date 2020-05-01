import numpy as np
from sklearn.svm import SVC  # noqa
from sklearn.model_selection import ShuffleSplit  # noqa
from mne.decoding import CSP  # noqa


def train_svm(epochs, labels):
    n_components = 3  # pick some components
    svc = SVC(C=1, kernel='linear')
    csp = CSP(n_components=n_components, norm_trace=False)

    # Define a monte-carlo cross-validation generator (reduce variance):
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    scores = []
    epochs_data = epochs.get_data()

    for train_idx, test_idx in cv.split(labels):
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = csp.fit_transform(epochs_data[train_idx], y_train)
        X_test = csp.transform(epochs_data[test_idx])

        # fit classifier
        svc.fit(X_train, y_train)

        scores.append(svc.score(X_test, y_test))

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                              class_balance))

    # Or use much more convenient scikit-learn cross_val_score function using
    # a Pipeline
    from sklearn.pipeline import Pipeline  # noqa
    from sklearn.model_selection import cross_val_score  # noqa
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    clf = Pipeline([('CSP', csp), ('SVC', svc)])
    scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)
    print(scores.mean())  # should match results above

    # And using reuglarized csp with Ledoit-Wolf estimator
    csp = CSP(n_components=n_components, reg='ledoit_wolf', norm_trace=False)
    clf = Pipeline([('CSP', csp), ('SVC', svc)])
    scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)

    print(scores.mean())  # should get better results than above

    return scores
