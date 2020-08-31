import pandas as pd

def import_Xy(path_train="./../../Data/raw_data/2243_9243_bundle_archive/fashion-mnist_train.csv",path_test="./../../Data/raw_data/2243_9243_bundle_archive/fashion-mnist_test.csv",label_name="label"):
    # importng the data from the paths which are there by default
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)
    X_train,y_train = df_train.drop("label",axis=1),df_train["label"]
    X_test,y_test = df_test.drop("label",axis=1),df_test["label"]
    return X_train,y_train, X_test,y_test

if __name__=="__main__":
    X_train, y_train, X_test, y_test = import_Xy()
    print("imported data")