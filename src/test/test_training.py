import os
import sys
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
    print(list(df))

    for l in [ 'haze', 'partly_cloudy', 'primary', 'road' ]:
        assert(l in df)

def test_image_features():
    image = rf.load_image("train_0")
    assert(len(image[0]) == 256)
    assert(len(image) == 256)
    assert(len(image[0][0]) == 3)
