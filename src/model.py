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
from skimage import io

PLANET_KAGGLE_ROOT = os.path.abspath("./input/")
PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')

labels_df = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)
# labels_df.head()

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


# print(labels_df)
# add_label_columns(labels_df)
# print(labels_df)

def load_image(image_name):
    return io.imread('./input/train-jpg/' + image_name + '.jpg')

IPython.embed()

# add new column to dataframe for weather class - single value 0,1,2,3
# weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']
# need new function image_to_feature_vector
    # given a sk image give me the flat array of values
# 
