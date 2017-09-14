# Read image data
# convert image chanels to numeric values
# flatten matrix into long vectors (concatenate back to back?)
# read classes per example to determine labels
# create classifer
# train
# evaluate
# along the way write helper functions to automate evaluation parts
import os
import pandas as pd
import IPython
import numpy as np
from skimage import io
from sklearn import datasets
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import svm

PLANET_KAGGLE_ROOT = os.path.abspath("./input/")
PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')

examples = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)

def labels_set(examples):
  labels = set()
  for tag_str in examples.tags.values:
    labels = labels.union(set(tag_str.split(' ')))
  labels = sorted(list(labels))
  return labels

def add_label_columns(df):
  labels = labels_set(df)
  for label in labels:
    df[label] = df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)


# print(labels_df) # add_label_columns(labels_df) # print(labels_df)

def load_image(image_name):
  return io.imread('./input/train-jpg/' + image_name + '.jpg')

def image_to_feature_vector(image):
    return image.flatten()

def feature_vector(image_name):
    return image_to_feature_vector(load_image(image_name))

def add_feature_vectors(df):
    df["feature_vector"] = df['image_name'].apply(feature_vector)

def weather_class(tags):
    classes  = ['clear', 'cloudy', 'haze', 'partly_cloudy']
    for (label,tag) in enumerate(classes):
        if tag in tags:
            return label

def add_weather_labels(df):
    df["weather_class"] = df['tags'].apply(weather_class)

from sklearn import datasets
from sklearn import svm

def run_thing():
    examples = pd.read_csv('./input/train_medium.csv')
    add_feature_vectors(examples)
    add_weather_labels(examples)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)

    #TODO: make this stuff work (probably have to convert to numpy ndarray or something)
    # label_matrix = examples['weather_class'].values.reshape(len(examples['weather_class']), 1)
    feature_matrix = np.matrix(examples['feature_vector'].tolist())
    clf.fit(feature_matrix, examples['weather_class'])
    print(clf.predict(feature_matrix[0:30]))

    clf2 = svm.SVC(gamma=0.001, C=100.)
    clf2.fit(feature_matrix, examples['weather_class'])
    print(clf2.predict(feature_matrix[0:30]))

if __name__ == "__main__":
    run_thing()

# 0
# .x x  A
# .x x  B
# .x x  C
# .x x  D
# .x x
# .
# 256*256*3
# OpenBLAS


# Run training

# Read csv
# Add feature_vector column
# Add weather_labels column
# Do sklearn stuff...


# image_name   feature_vector   weather _class   tags
# 001          [......]          0               agriculture clear habitation primary road
# 002          [......]          2               haze primary
# 003          [......]          0               haze primary

# IPython.embed()
# add new column to dataframe for weather class - single value 0,1,2,3
# produce 1 number for each row representing that row's class
# weather_labels = ['clear', 'cloudy', 'haze', 'partly_cloudy']
# need new function image_to_feature_vector
# given a sk image give me the flat array of values


# clf.fit(df[]"feaure_vector"], df["classes"])
