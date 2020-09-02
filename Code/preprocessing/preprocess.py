import pandas as pd
import numpy as np
from PIL import Image
import os


PATH_TRAIN = "../../Data/raw/fashion-mnist_train.csv"
PATH_TEST = "../../Data/raw/fashion-mnist_test.csv"
DATA_PATH = "../../Data/raw/"

dict_fashion = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


def csv2img(csv, path, is_train=True):
    """
    Convert pixel values from .csv to .png image
    Source: https://www.kaggle.com/alexanch/image-classification-w-fastai-fashion-mnist
    """
    # define the name of the directory to be created
    if is_train:
        image_path = "working/train/"
    else:
        image_path = "working/test/"
    full_path = os.path.join(path, image_path)
    try:
        os.makedirs(full_path)
    except OSError:
        print("Creation of the directory %s failed" % full_path)
    else:
        print("Successfully created the directory %s" % full_path)

    if not os.path.isdir(full_path):
        for i in range(len(csv)):
            # csv.iloc[i, 1:].to_numpy() returns pixel values array
            # for i'th imag excluding the label
            # next step: reshape the array to original shape(28, 28)
            # and add missing color channels
            result = Image.fromarray(np.uint8(
                np.stack(
                    np.rot90(
                        csv.iloc[i, 1:].to_numpy().
                        reshape((28, 28)))*3, axis=-1)))
            # save the image:
            result.save(f'{full_path}{str(i)}.png')

        print(f'{len(csv)} images were created.')


def create_train_test(csv_train, csv_test, data_path=DATA_PATH):
    """Create images on `data_path` from data provided by csvs.
    This is just a wrapper of csv2img to create the images provided
    by many csvs at once.

    Args:
        csv_list ([type]): [description]
        data_path (str, optional): [description]. Defaults to "../../Data/raw".
    """
    csv2img(csv_train, data_path, True)
    csv2img(csv_test, data_path, False)


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

    # creating images from csv data
    create_train_test(df_train, df_test)

    # creating labels
    df_train['label_text'] = df_train['label'].apply(lambda x: dict_fashion[x])
    df_test['label_text'] = df_test['label'].apply(lambda x: dict_fashion[x])

    # add image names:
    df_train['img'] = pd.Series([str(i)+'.png' for i in range(len(df_train))])
    df_test['img'] = pd.Series([str(i)+'.png' for i in range(len(df_test))])
    X_train, y_train = df_train.drop("label", axis=1), df_train["label"]
    X_test, y_test = df_test.drop("label", axis=1), df_test["label"]

    # save corresponding labels and image names to .csv file:
    df_train[['img', 'label_text']].to_csv(
        os.path.join(DATA_PATH,
                     'working/train/train_image_labels.csv'), index=False)

    df_test[['img', 'label_text']].to_csv(
        os.path.join(DATA_PATH,
                     'working/test/test_image_labels.csv'), index=False)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = import_xy()
    print("imported data")
