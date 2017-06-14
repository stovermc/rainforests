import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rainforests.core as rf

def test_collecting_labels():
  labels = rf.labels_set(rf.examples)

  for l in ['agriculture', 'water', 'primary', 'road']:
    assert(l in labels)

def test_add_label_columns():
    df = rf.examples
    list(df)
    assert(['image_name', 'tags']) == list(df)

    rf.add_label_columns(df)

    for l in [ 'haze', 'partly_cloudy', 'primary', 'road' ]:
        assert(l in df)

def test_image_features():
    image = rf.load_image("train_0")
    assert(len(image[0]) == 256)
    assert(len(image) == 256)
    assert(len(image[0][0]) == 3)

def test_image_to_feature_vector():
    example = np.array([[["r", "g", "b"], ["r", "g", "b"]],
               [["r", "g", "b"], ["r", "g", "b"]]])
    example_vector = rf.image_to_feature_vector(example)
    assert(["r", "g", "b", "r", "g", "b", "r", "g", "b", "r", "g", "b"])
    image = rf.load_image("train_0")
    pixel_vector = rf.image_to_feature_vector(image)
    assert(len(pixel_vector)== 256 * 256 * 3)

def test_feature_vector_to_classes():
    # examples = rf.examples
    examples = pd.read_csv('./input/test.csv')
    rf.add_feature_vectors(examples)
    assert('feature_vector' in examples)
