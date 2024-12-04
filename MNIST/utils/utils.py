import logging
import os
import random
import time

import colorlog
import torch
import numpy as np

from utils.parameters import Params


def record_time(params: Params, t=None, name=None):
    if t and name and params.save_timing == name or params.save_timing is True:
        torch.cuda.synchronize()
        params.timing_data[name].append(round(1000 * (time.perf_counter() - t)))


def create_table(params: dict):
    data = "| name | value | \n |-----|-----|"

    for key, value in params.items():
        data += '\n' + f"| {key} | {value} |"

    return data

def create_logger():
    """
        Setup the logging environment
    """
    log = logging.getLogger()  # root logger
    log.setLevel(logging.DEBUG)
    format_str = '%(asctime)s - %(levelname)-8s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    if os.isatty(2):
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'reset',
                  'INFO': 'reset',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_red'}
        formatter = colorlog.ColoredFormatter(cformat, date_format,
                                              log_colors=colors)
    else:
        formatter = logging.Formatter(format_str, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return logging.getLogger(__name__)


def th(vector):
    return torch.tanh(vector) / 2 + 0.5

def model2vector(model):
    nparr = np.array([])
    for key, var in model.items():
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr = np.append(nparr, nplist)
    return nparr


def norm_scale(nparr1, nparr2):
    vnum = np.linalg.norm(nparr1, ord=None, axis=None, keepdims=False) + 1e-9
    return vnum / np.linalg.norm(nparr2, ord=None, axis=None, keepdims=False) + 1e-9
