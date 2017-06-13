import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rainforests.core as rf

def test_collecting_labels():
  labels = rf.labels_set(rf.labels_df)

  for l in ['agriculture', 'water', 'primary', 'road']:
    assert(l in labels)
