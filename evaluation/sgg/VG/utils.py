# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import bisect
import copy
import logging

import json
import torch
import torch.utils.data


from .imports import import_file


from . import datasets as D


# by Jiaxin
def get_dataset_statistics(cfg):
    """
    get dataset statistics (e.g., frequency bias) from training data
    will be called to help construct FrequencyBias module
    """
    logger = logging.getLogger(__name__)
    logger.info('-'*100)
    logger.info('get dataset statistics...')
    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_names = cfg.DATASETS.TRAIN

    data_statistics_name = ''.join(dataset_names) + '_statistics'
    save_file = os.path.join(cfg.OUTPUT_DIR, "{}.cache".format(data_statistics_name))
    
    if os.path.exists(save_file):
        logger.info('Loading data statistics from: ' + str(save_file))
        logger.info('-'*100)
        return torch.load(save_file, map_location=torch.device("cpu"))
    else:
        logger.info('Unable to load data statistics from: ' + str(save_file))

    statistics = []
    for dataset_name in dataset_names:
        data = DatasetCatalog.get(dataset_name, cfg)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # Remove it because not part of the original repo (factory cant deal with additional parameters...).
        if "capgraphs_file" in args.keys():
            del args["capgraphs_file"]
        dataset = factory(**args)
        statistics.append(dataset.get_statistics())
    logger.info('finish')

    assert len(statistics) == 1
    result = {
        'fg_matrix': statistics[0]['fg_matrix'],
        'pred_dist': statistics[0]['pred_dist'],
        'obj_classes': statistics[0]['obj_classes'], # must be exactly same for multiple datasets
        'rel_classes': statistics[0]['rel_classes'],
        'att_classes': statistics[0]['att_classes'],
    }
    logger.info('Save data statistics to: ' + str(save_file))
    logger.info('-'*100)
    torch.save(result, save_file)
    return result
