import pandas as pd

PATH_TRAIN = "../../Data/raw/fashion-mnist_train.csv"
PATH_TEST = "../../Data/raw/fashion-mnist_test.csv"


def import_xy(
        path_train=PATH_TRAIN,
        path_test=PATH_TEST,
        label_name="label"):
    """Import data from specified path.

    Args:
        path_train (str, optional): [description]. Defaults to PATH_TRAIN.
        path_test ([type], optional): [description]. Defaults to PATH_TEST.
        label_name (str, optional): [description]. Defaults to "label".

    Returns:
        [type]: [description]
    """
    # importng the data from the paths which are there by default
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)
    X_train, y_train = df_train.drop("label", axis=1), df_train["label"]
    X_test, y_test = df_test.drop("label", axis=1), df_test["label"]
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = import_xy()
    print("imported data")
