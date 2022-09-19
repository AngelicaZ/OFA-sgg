import json
import sys
import os.path as op
import numpy as np


def evaluate_on_coco_caption(res_file, label_file, outfile=None):
    """
    res_file: txt file, each row is [image_key, json format list of captions].
             Each caption is a dict, with fields "caption", "conf".
    label_file: JSON file of ground truth captions in GQA format.
    """
    pass

if __name__ == "__main__":
    if len(sys.argv) == 3:
        evaluate_on_coco_caption(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        evaluate_on_coco_caption(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        raise NotImplementedError