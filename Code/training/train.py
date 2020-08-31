preprocess_path = "./../preprocessing"
import sys
sys.path.append(preprocess_path)
from preprocess import import_Xy
import joblib
import logging

def train_model():
    X_train,y_train,X_test,y_test = import_Xy()
    from sklearn.svm import SVC
    model = SVC(kernel='linear')
    model.fit(X_train, y_trainx)
    from sklearn.metrics import precision_score
    y_pred = model.predict(X_test)
    precision = precision_score(y_test,y_pred,average='weighted')

    return model,precision 

def save_model(model,precision,model_path = "./../../Data/models/latest_model.pkl"):
    
    print(precision)
    joblib.dump(model,model_path)

if __name__ == "__main__":
    model, precision = train_model()
    save_model(model, precision)
 