from sklearn.decomposition.pca import PCA
from sklearn.metrics import precision_score, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import logging
import joblib
import sys
sys.path.append("..")
from preprocessing.preprocess import import_xy

MODEL_PATH = "../../Data/models/latest_model.pkl"


def train_model():
    X_train, y_train, X_test, y_test = import_xy()
    X_train /= 255.0
    X_test /= 255.0

    # model = SVC(kernel='linear')
    # model = make_pipeline(PCA(200),
    #                      LGBMClassifier(silent=False, random_state=42))
    model = MLPClassifier((128, 64, 8), verbose=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = accuracy_score(y_test, y_pred, average='weighted')
    print(classification_report(y_test, y_pred))
    return model, precision


def save_model(model, precision, model_path=MODEL_PATH):

    print(precision)
    joblib.dump(model, model_path)


if __name__ == "__main__":
    model, precision = train_model()
    save_model(model, precision)
    print("Success!")
