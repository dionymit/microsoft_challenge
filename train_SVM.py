def train_svm(train_csv, label_csv):
    """
    @Author: Xu Yan
        @Edits: Dionysis Mitsios

    Description: Used to retrain the rf model

    Arguments:
    train_csv -- where the train CSV features are saved
    label_csv -- where the annotations CSV are saved
    """
    import numpy as np
    import pickle
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Load training data X, y
    features_merged_train = np.loadtxt(train_csv, delimiter=",")
    labels_train = np.loadtxt(label_csv, delimiter=",")

    # Train/fit
    sc = StandardScaler()
    svm = SVC(degree=1, gamma="scale", random_state=42, C=4, kernel="poly")
    clf = Pipeline(steps=[("scaler", sc), ("svm", svm)]).fit(
        features_merged_train, labels_train
    )

    # Save using pickle
    with open("model_svm_new.pkl", "wb") as f:
        pickle.dump(clf, f)
