import os
from feature_extractor.reid_config import cfg
from .baseline.model import make_model


def build_reid_model(_mcmt_cfg):

    #  slices the string abs_file from the beginning up to (but not including) the last forward slash. 
    # the : refers to Start:End, so this means from start: end is the last slash position

    cfg.INPUT.SIZE_TEST = _mcmt_cfg.reid_config.reid_size_test
    cfg.MODEL.NAME = _mcmt_cfg.backbone
    model = make_model(cfg, num_class=100)
    model.load_param(_mcmt_cfg.reid_config.reid_model_path)

    return model,cfg
