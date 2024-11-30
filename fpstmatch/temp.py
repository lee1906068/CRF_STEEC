import copy
import datetime
import math
import os
import pprint
import random
import time
import warnings
from ast import literal_eval
from datetime import datetime, timedelta
from itertools import product
from multiprocessing import Pool

import fpstmatch as fp
import geopandas as gpd
import geopy.distance
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import psycopg2
import torch
import transbigdata as tbd
from geopy.distance import geodesic
from pandarallel import pandarallel
from shapely import unary_union
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from . import basicfunc, creategraph, logs, stats, util_coord, utils_graph


def _pe_lp_i(lp_path_basicvar, st_eulid_dur, pe_lp_arg, K_max):
    a1 = pe_lp_arg["a1"]
    a21 = pe_lp_arg["a21"]
    a22 = pe_lp_arg["a22"]
    a31 = pe_lp_arg["a31"]
    a32 = pe_lp_arg["a32"]
    a33 = pe_lp_arg["a33"]
    b1 = pe_lp_arg["b1"]
    b2 = pe_lp_arg["b2"]
    b3 = pe_lp_arg["b3"]

    _Epe_lp_list = list()
    for i in range(0, len(lp_path_basicvar)):
        i_len_eulid = st_eulid_dur[i, :][0]

        i_lp_len = np.delete(
            lp_path_basicvar[i][:, 0], np.where(lp_path_basicvar[i][:, 0] == -1)
        )
        i_lp_speed = np.delete(
            lp_path_basicvar[i][:, 1], np.where(lp_path_basicvar[i][:, 1] == -1)
        )
        i_lp_dir_head = np.delete(
            lp_path_basicvar[i][:, 2], np.where(lp_path_basicvar[i][:, 2] == -1)
        )
        i_lp_dir_turn = np.delete(
            lp_path_basicvar[i][:, 3], np.where(lp_path_basicvar[i][:, 3] == -1)
        )
        i_lp_det_count = np.delete(
            lp_path_basicvar[i][:, 4], np.where(lp_path_basicvar[i][:, 4] == -1)
        )
        i_lp_det_dist = np.delete(
            lp_path_basicvar[i][:, 5], np.where(lp_path_basicvar[i][:, 5] == -1)
        )
        i_lp_det_len = np.delete(
            lp_path_basicvar[i][:, 6], np.where(lp_path_basicvar[i][:, 6] == -1)
        )

        var_lp_len = 1 - (
            abs(i_lp_len - i_len_eulid) - abs(i_lp_len - i_len_eulid).min()
        ) / (abs(i_lp_len - i_len_eulid).max() - abs(i_lp_len - i_len_eulid).min())
        _pe_Llp = var_lp_len / var_lp_len.sum()
        _Epe_lp_len = -_pe_Llp * np.log(_pe_Llp)

        var_lp_dir_head = 1 - (i_lp_dir_head - i_lp_dir_head.min()) / (
            i_lp_dir_head.max() - i_lp_dir_head.min()
        )
        _pe_Hlp = var_lp_dir_head / var_lp_dir_head.sum()
        _Epe_lp_dir_head = np.nan_to_num(-_pe_Hlp * np.log(_pe_Hlp))

        var_lp_dir_trun = 1 - (i_lp_dir_turn - i_lp_dir_turn.min()) / (
            i_lp_dir_turn.max() - i_lp_dir_turn.min()
        )
        _pe_Tlp = var_lp_dir_trun / var_lp_dir_trun.sum()
        _Epe_lp_dir_trun = np.nan_to_num(-_pe_Tlp * np.log(_pe_Tlp))

        var_lp_det_count = 1 - (i_lp_det_count - i_lp_det_count.min()) / (
            i_lp_det_count.max() - i_lp_det_count.min()
        )
        _pe_Clp = var_lp_det_count / var_lp_det_count.sum()
        _Epe_lp_det_count = np.nan_to_num(-_pe_Clp * np.log(_pe_Clp))

        var_lp_det_dist = 1 - (i_lp_det_dist - i_lp_det_dist.min()) / (
            i_lp_det_dist.max() - i_lp_det_dist.min()
        )
        _pe_DDlp = var_lp_det_dist / var_lp_det_dist.sum()
        _Epe_lp_det_dist = np.nan_to_num(-_pe_DDlp * np.log(_pe_DDlp))

        var_lp_det_len = 1 - (i_lp_det_len - i_lp_det_len.min()) / (
            i_lp_det_len.max() - i_lp_det_len.min()
        )
        _pe_LDlp = var_lp_det_len / var_lp_det_len.sum()
        _Epe_lp_det_len = np.nan_to_num(-_pe_LDlp * np.log(_pe_LDlp))

        _Epe_lp_dir = a21 * _Epe_lp_dir_head + a22 * _Epe_lp_dir_trun
        _Epe_lp_det = (
            a31 * _Epe_lp_det_count + a32 * _Epe_lp_det_dist + a33 * _Epe_lp_det_len
        )

        _Epe_lp = b1 * _Epe_lp_len + b2 * _Epe_lp_dir + b3 * _Epe_lp_det

        _Epe_lp = np.pad(
            _Epe_lp, (0, K_max - len(_Epe_lp)), "constant", constant_values=0
        )
        _Epe_lp_len = np.pad(
            _Epe_lp_len, (0, K_max - len(_Epe_lp_len)), "constant", constant_values=-1
        )
        _Epe_lp_dir = np.pad(
            _Epe_lp_dir, (0, K_max - len(_Epe_lp_dir)), "constant", constant_values=-1
        )
        _Epe_lp_det = np.pad(
            _Epe_lp_det, (0, K_max - len(_Epe_lp_det)), "constant", constant_values=-1
        )
        _Epe_lp_dir_head = np.pad(
            _Epe_lp_dir_head,
            (0, K_max - len(_Epe_lp_dir_head)),
            "constant",
            constant_values=-1,
        )
        _Epe_lp_dir_trun = np.pad(
            _Epe_lp_dir_trun,
            (0, K_max - len(_Epe_lp_dir_trun)),
            "constant",
            constant_values=-1,
        )
        _Epe_lp_det_count = np.pad(
            _Epe_lp_det_count,
            (0, K_max - len(_Epe_lp_det_count)),
            "constant",
            constant_values=-1,
        )
        _Epe_lp_det_dist = np.pad(
            _Epe_lp_det_dist,
            (0, K_max - len(_Epe_lp_det_dist)),
            "constant",
            constant_values=-1,
        )
        _Epe_lp_det_len = np.pad(
            _Epe_lp_det_len,
            (0, K_max - len(_Epe_lp_det_len)),
            "constant",
            constant_values=-1,
        )
        _Epe_lp_list.append(_Epe_lp)
        """
        _Epe_lp_list.append(
            np.stack(
                [
                    _Epe_lp,
                    _Epe_lp_len,
                    _Epe_lp_dir,
                    _Epe_lp_det,
                    _Epe_lp_dir_head,
                    _Epe_lp_dir_trun,
                    _Epe_lp_det_count,
                    _Epe_lp_det_dist,
                    _Epe_lp_det_len,
                ],
                axis=1,
            )
        )
        """
    _Epe_lp_var = np.nan_to_num(np.stack(_Epe_lp_list, axis=1))
    return _Epe_lp_var


def pe_lp_i(lp_path_basicvar, st_eulid_dur, pe_lp_arg, K_max):

    a1 = pe_lp_arg["a1"]
    a21 = pe_lp_arg["a21"]
    a22 = pe_lp_arg["a22"]
    a31 = pe_lp_arg["a31"]
    a32 = pe_lp_arg["a32"]
    a33 = pe_lp_arg["a33"]
    b1 = pe_lp_arg["b1"]
    b2 = pe_lp_arg["b2"]
    b3 = pe_lp_arg["b3"]

    _Epe_lp_list = list()
    for i in range(0, len(lp_path_basicvar)):
        i_len_eulid = st_eulid_dur[i, :][0]

        i_lp_len = np.delete(
            lp_path_basicvar[i][:, 0], np.where(lp_path_basicvar[i][:, 0] == -1)
        )
        i_lp_speed = np.delete(
            lp_path_basicvar[i][:, 1], np.where(lp_path_basicvar[i][:, 1] == -1)
        )
        i_lp_dir_head = np.delete(
            lp_path_basicvar[i][:, 2], np.where(lp_path_basicvar[i][:, 2] == -1)
        )
        i_lp_dir_turn = np.delete(
            lp_path_basicvar[i][:, 3], np.where(lp_path_basicvar[i][:, 3] == -1)
        )
        i_lp_det_count = np.delete(
            lp_path_basicvar[i][:, 4], np.where(lp_path_basicvar[i][:, 4] == -1)
        )
        i_lp_det_dist = np.delete(
            lp_path_basicvar[i][:, 5], np.where(lp_path_basicvar[i][:, 5] == -1)
        )
        i_lp_det_len = np.delete(
            lp_path_basicvar[i][:, 6], np.where(lp_path_basicvar[i][:, 6] == -1)
        )

        var_lp_len = 1 - (
            abs(i_lp_len - i_len_eulid) - abs(i_lp_len - i_len_eulid).min()
        ) / (abs(i_lp_len - i_len_eulid).max() - abs(i_lp_len - i_len_eulid).min())
        var_lp_len = np.nan_to_num(var_lp_len)

        var_lp_dir_head = 1 - (i_lp_dir_head - i_lp_dir_head.min()) / (
            i_lp_dir_head.max() - i_lp_dir_head.min()
        )
        var_lp_dir_trun = 1 - (i_lp_dir_turn - i_lp_dir_turn.min()) / (
            i_lp_dir_turn.max() - i_lp_dir_turn.min()
        )
        _var_lp_dir = a21 * var_lp_dir_head + a22 * var_lp_dir_trun
        #var_lp_dir = (_var_lp_dir - _var_lp_dir.min()) / (
        #    _var_lp_dir.max() - _var_lp_dir.min()
        #)
        var_lp_dir = _var_lp_dir/2
        var_lp_dir = np.nan_to_num(var_lp_dir)

        var_lp_det_count = 1 - (i_lp_det_count - i_lp_det_count.min()) / (
            i_lp_det_count.max() - i_lp_det_count.min()
        )
        var_lp_det_dist = 1 - (i_lp_det_dist - i_lp_det_dist.min()) / (
            i_lp_det_dist.max() - i_lp_det_dist.min()
        )
        var_lp_det_len = 1 - (i_lp_det_len - i_lp_det_len.min()) / (
            i_lp_det_len.max() - i_lp_det_len.min()
        )
        _var_lp_det = (
            a31 * var_lp_det_count + a32 * var_lp_det_dist + a33 * var_lp_det_len
        )
        #var_lp_det = (_var_lp_det - _var_lp_det.min()) / (
        #    _var_lp_det.max() - _var_lp_det.min()
        #)
        var_lp_det = _var_lp_det/3
        var_lp_det = np.nan_to_num(var_lp_det)

        _pe_lp = b1 * var_lp_len + b2 * var_lp_dir + b3 * var_lp_det
        pe_lp = _pe_lp / _pe_lp.sum()
        _Epe_lp = np.nan_to_num(-pe_lp * np.log(pe_lp))

        _Epe_lp = np.pad(
            _Epe_lp, (0, K_max - len(_Epe_lp)), "constant", constant_values=-1
        )

        _Epe_lp_list.append(_Epe_lp)

    Epe_lp = np.nan_to_num(np.stack(_Epe_lp_list, axis=1))
    return Epe_lp


def getST_v_mu_sigma_init(Epe_lp, lp_path_speed):
    _lp_v_mu = list()
    _lp_v_sigma = list()
    _argmax_k_init = list()
    for i in range(0, Epe_lp.shape[1]):
        #i_pe_lp = np.delete(Epe_lp[:, i], np.where(Epe_lp[:, i] == -1))
        i_pe_lp = Epe_lp[:, i]
        _argmax_k = np.argmax(i_pe_lp)
        _argmax_k_init.append(_argmax_k)
        _lp_v_mu.append(lp_path_speed[:, i][_argmax_k])
        _lp_v_sigma.append(
            np.std(np.delete(lp_path_speed[:, i], np.where(lp_path_speed[:, i] == -1)))
        )
    return (
        _argmax_k_init,
        _lp_v_mu,
        _lp_v_sigma,
        np.mean(_lp_v_mu),
        np.std(_lp_v_mu),
    )


def getST_v_mu_sigma(lp_path_speed, argmax_k):

    _lp_v_mu = list()
    _lp_v_sigma = list()
    for i, _argmax_k in enumerate(argmax_k):
        _lp_v_mu.append(lp_path_speed[:, i][_argmax_k])
        _lp_v_sigma.append(
            np.std(np.delete(lp_path_speed[:, i], np.where(lp_path_speed[:, i] == -1)))
        )

    return (
        _lp_v_mu,
        _lp_v_sigma,
        np.mean(_lp_v_mu),
        np.std(_lp_v_mu),
    )

def update_lp_v_mu_sigma_ab(
    segE, lp_path_speed, lp_v_mu, lp_v_sigma, ST_v_mu, ST_v_sigma, pt_lp_arg
):

    a = pt_lp_arg["a"]
    eta = pt_lp_arg["eta"]
    alpha = pt_lp_arg["alpha"]

    _lp_path_speed = lp_path_speed.copy()
    _lp_v_mu = lp_v_mu.copy()
    _lp_v_sigma = lp_v_sigma.copy()

    # lp_v_ab_is = np.where((lp_v_mu < ST_v_mu - (alpha * ST_v_sigma)))[0]
    lp_v_ab_is = np.where(
        (lp_v_mu < ST_v_mu - (alpha * ST_v_sigma)) | (lp_v_mu > ST_v_mu + (alpha * ST_v_sigma))
    )[0]

    if len(lp_v_ab_is) == 0:
        return lp_path_speed, lp_v_mu, lp_v_sigma, ST_v_mu, ST_v_sigma

    _sum = 0
    itr = 0

    _segE = np.pad(segE, (a, a), "constant", constant_values=0)

    while True:
        __lp_v_mu = _lp_v_mu.copy()
        _lp_v_mu_ = np.pad(_lp_v_mu, (a, a), "constant", constant_values=0)
        _lp_v_sigma_ = np.pad(_lp_v_sigma, (a, a), "constant", constant_values=0)
        itr += 1
        for i, lp_v_ab_i in enumerate(lp_v_ab_is):
            _segE_a = _segE[lp_v_ab_i : lp_v_ab_i + 2 * a + 1]
            _lp_v_mu_a = _lp_v_mu_[lp_v_ab_i : lp_v_ab_i + 2 * a + 1]
            _lp_v_sigma__ = _lp_v_sigma_[lp_v_ab_i : lp_v_ab_i + 2 * a + 1]
            _lp_v_mu_lp_v_ab_i = 0
            _lp_v_sigma_lp_v_ab_i = 0
            for j in range(1, a + 1):
                i_v = _segE_a[a - j] / (_segE_a.sum() - _segE_a[a]) * _lp_v_mu_a[a - j]

                iv_ = _segE_a[a + j] / (_segE_a.sum() - _segE_a[a]) * _lp_v_mu_a[a + j]

                _lp_v_mu_lp_v_ab_i += (
                    eta[j - 1] * (i_v + iv_)
                    if (i_v != 0) and (iv_ != 0)
                    else eta[j - 1] * 2 * (_lp_v_mu_a[a - j] + _lp_v_mu_a[a + j])
                )

                i_sigma = (
                    _segE_a[a - j] / (_segE_a.sum() - _segE_a[a]) * _lp_v_sigma__[a - j]
                )

                isigma_ = (
                    _segE_a[a + j] / (_segE_a.sum() - _segE_a[a]) * _lp_v_sigma__[a + j]
                )

                _lp_v_sigma_lp_v_ab_i += (
                    eta[j - 1] * (i_sigma + isigma_)
                    if (i_sigma != 0) and (i_sigma != 0)
                    else eta[j - 1] * 2 * (_lp_v_sigma__[a - j] + _lp_v_sigma__[a + j])
                )

            _lp_v_mu[lp_v_ab_i] = _lp_v_mu_lp_v_ab_i
            _lp_v_sigma[lp_v_ab_i] = _lp_v_sigma_lp_v_ab_i

            itr_sigma_control = True
            while itr_sigma_control:
                _itr_sigma = _lp_v_sigma[lp_v_ab_i]
                _lp_path_speed[:, lp_v_ab_i][
                    np.where(
                        (
                            lp_path_speed[:, lp_v_ab_i]
                            < _lp_v_mu_lp_v_ab_i - (alpha * _lp_v_sigma[lp_v_ab_i])
                        )
                        & (lp_path_speed[:, lp_v_ab_i] != -1)
                    )[0]
                ] = _lp_v_mu_lp_v_ab_i

                _lp_path_speed[:, lp_v_ab_i][
                    np.where(
                        (
                            lp_path_speed[:, lp_v_ab_i]
                            > _lp_v_mu_lp_v_ab_i + (alpha * _lp_v_sigma[lp_v_ab_i])
                        )
                        & (lp_path_speed[:, lp_v_ab_i] != -1)
                    )[0]
                ] = _lp_v_mu_lp_v_ab_i

                _lp_v_sigma[lp_v_ab_i] = (
                    np.std(
                        np.delete(
                            lp_path_speed[:, lp_v_ab_i],
                            np.where(lp_path_speed[:, lp_v_ab_i] == -1),
                        )
                    )
                    + _lp_v_sigma_lp_v_ab_i
                )
                if _itr_sigma == _lp_v_sigma[lp_v_ab_i]:
                    itr_sigma_control = False

        _ST_v_mu = np.mean(_lp_v_mu)
        _ST_v_sigma = np.std(_lp_v_mu)

        lp_v_ab_is = np.where(
             (_lp_v_mu < _ST_v_mu - (alpha * ST_v_sigma)) | (_lp_v_mu > _ST_v_mu + (alpha * ST_v_sigma))
            )[0]
        itr += 1
        if len(lp_v_ab_is) == 0:
            # logs.info(f"update_lp_v_mu_sigma_ab itr={itr}")

            return _lp_path_speed, _lp_v_mu, _lp_v_sigma, _ST_v_mu, _ST_v_sigma

        if abs(np.array(__lp_v_mu) - np.array(_lp_v_mu)).sum() == _sum:
            # logs.info(f"update_lp_v_mu_sigma_ab itr={itr}")

            return _lp_path_speed, _lp_v_mu, _lp_v_sigma, _ST_v_mu, _ST_v_sigma

        _sum = abs(np.array(__lp_v_mu) - np.array(_lp_v_mu)).sum()


"""
get segE of ST_seg

"""


def getsegE(segE_var, segE_arg):

    st_eulid = segE_var["st_eulid"]
    st_dur = segE_var["st_dur"]
    delta_L = segE_var["delta_L"]
    delta_T = segE_var["delta_T"]

    w1 = segE_arg["w1"]
    w2 = segE_arg["w2"]
    segE_len = -st_eulid / delta_L * np.log(st_eulid / delta_L)
    segE_dur = -st_dur / delta_T * np.log(st_dur / delta_T)
    segE = w1 * segE_dur + w2 * segE_len

    return segE


def pt_lp_i_afb(pt_lp_var, pt_lp_arg):
    segE = pt_lp_var["segE"]
    lp_path_speed = pt_lp_var["lp_path_speed"]
    ST_v_sigma = pt_lp_var["ST_v_sigma"]
    K_max = pt_lp_var["K_max"]

    a = pt_lp_arg["a"]
    eta = pt_lp_arg["eta"]

    if len(eta) != a:
        print("args error")

    _segE = np.pad(segE, (a, a), "constant", constant_values=0)
    _lp_path_speed = np.pad(
        lp_path_speed, ((a, a), (0, 0)), "constant", constant_values=0
    )

    Ept_lp_ = list()
    for i in range(len(lp_path_speed)):
        _segE_a = _segE[i : i + 2 * a + 1]
        _lp_path_speed_a = _lp_path_speed[i : i + 2 * a + 1]

        if np.argmax(_segE_a) == a:
            _speed_i = np.delete(
                _lp_path_speed_a[a], np.where(_lp_path_speed_a[a] == -1)
            )
            shape_pt_i = (
                (
                    (_lp_path_speed_a[a] != -1).sum(),
                    (_lp_path_speed_a[a - 1] != -1).sum(),
                )
                if i > 0
                else (
                    (_lp_path_speed_a[a] != -1).sum(),
                    (_lp_path_speed_a[a] != -1).sum(),
                )
            )
            _pt_lp_i = np.ones(shape_pt_i) * (
                1 / (np.std(_speed_i) * np.sqrt(2 * np.pi))
            )

            _Ept_lp_i = -_pt_lp_i * np.log(_pt_lp_i)

            shape_pt_i_pad = (
                (
                    (0, K_max - (_lp_path_speed_a[a] != -1).sum()),
                    (0, K_max - (_lp_path_speed_a[a - 1] != -1).sum()),
                )
                if i > 0
                else (
                    (0, K_max - (_lp_path_speed_a[a] != -1).sum()),
                    (0, K_max - (_lp_path_speed_a[a] != -1).sum()),
                )
            )

            _Ept_lp_i = np.pad(_Ept_lp_i, shape_pt_i_pad, "constant", constant_values=-1)
        else:
            _pt_lp_i = 0

            for j in range(1, a + 1):

                _speed_i = np.delete(
                    _lp_path_speed_a[a], np.where(_lp_path_speed_a[a] == -1)
                )
                _speed_i_v = (
                    np.delete(
                        _lp_path_speed_a[a - j], np.where(_lp_path_speed_a[a - j] == -1)
                    )
                    .reshape(1, -1)
                    .T
                    if i > 0
                    else _speed_i.reshape(1, -1).T
                )
                _speed_iv_ = (
                    np.delete(
                        _lp_path_speed_a[a + j], np.where(_lp_path_speed_a[a + j] == -1)
                    )
                    .reshape(1, -1)
                    .T
                    if i < len(lp_path_speed) - 1
                    else _speed_i.reshape(1, -1).T
                )

                segE_i_v = (
                    _segE_a[a - j]
                    / (_segE_a.sum() - _segE_a[a])
                    * fp.N(_speed_i, _speed_i_v, np.std(_speed_i_v))
                )
                segE_iv_ = (
                    _segE_a[a + j]
                    / (_segE_a.sum() - _segE_a[a])
                    * fp.N(_speed_i, _speed_iv_, np.std(_speed_iv_))
                )

                shape_pt_i_v_pad = (
                    (
                        (0, K_max - (_lp_path_speed_a[a - j] != -1).sum()),
                        (0, K_max - (_lp_path_speed_a[a] != -1).sum()),
                    )
                    if i > 0
                    else (
                        (0, K_max - (_lp_path_speed_a[a] != -1).sum()),
                        (0, K_max - (_lp_path_speed_a[a] != -1).sum()),
                    )
                )
                shape_pt_iv__pad = (
                    (
                        (0, K_max - (_lp_path_speed_a[a + j] != -1).sum()),
                        (0, K_max - (_lp_path_speed_a[a] != -1).sum()),
                    )
                    if i < len(lp_path_speed)
                    else (
                        (0, K_max - (_lp_path_speed_a[a] != -1).sum()),
                        (0, K_max - (_lp_path_speed_a[a] != -1).sum()),
                    )
                )

                segE_i_v = (
                    np.pad(segE_i_v, shape_pt_i_v_pad, "constant", constant_values=-1)
                    if _segE_a[a - j] != 0
                    else np.zeros((K_max, K_max))
                )
                segE_iv_ = (
                    np.pad(segE_iv_, shape_pt_iv__pad, "constant", constant_values=-1)
                    if _segE_a[a + j] != 0
                    else np.zeros((K_max, K_max))
                )

                _pt_lp_i += (
                    eta[j - 1] * 2 * (segE_i_v + segE_iv_)
                    if (_segE_a[a - j] != 0) and (_segE_a[a + j] != 0)
                    else eta[j - 1] * 2 * 2 * (segE_i_v + segE_iv_)
                )

            _Ept_lp_i = -_pt_lp_i * np.log(_pt_lp_i)

        Ept_lp_.append(np.nan_to_num(_Ept_lp_i))

    return np.nan_to_num(np.stack(Ept_lp_, axis=0))


def pt_lp_i_1step(pt_lp_var):
    segE = pt_lp_var["segE"]
    K_max = pt_lp_var["K_max"]
    lp_path_speed = pt_lp_var["lp_path_speed"]
    lp_v_sigma = pt_lp_var["lp_v_sigma"]
    ST_v_sigma = pt_lp_var["ST_v_sigma"]

    _Ept_lp = list()

    for i in range(len(segE)):
        if i == 0:
            _Ept_lp_i = np.zeros(
                (lp_path_speed[:, i].shape[0], lp_path_speed[:, i].shape[0])
            )
            _Ept_lp.append(np.nan_to_num(_Ept_lp_i))
        else:
            try:
                _speed_i = (
                    np.delete(lp_path_speed[:, i], np.where(lp_path_speed[:, i] == -1))
                    .reshape(1, -1)
                    .T
                )
            except:
                print(i, segE)
            _speed_i_1 = np.delete(
                lp_path_speed[:, i - 1], np.where(lp_path_speed[:, i - 1] == -1)
            )
            lp_v_sigma_i = lp_v_sigma[i]
            lp_v_sigma_i_1 = lp_v_sigma[i - 1]

            if segE[i] > segE[i - 1]:
                _pt_lp_i = basicfunc.N(_speed_i, _speed_i_1, lp_v_sigma_i)
            else:
                _pt_lp_i = basicfunc.N(_speed_i, _speed_i_1, lp_v_sigma_i_1)

            # _pt_lp_i = np.nan_to_num(_pt_lp_i / _pt_lp_i.sum())
            # _Ept_lp_i = -_pt_lp_i * np.log(_pt_lp_i)
            # _Ept_lp_i = - np.log(np.nan_to_num(_pt_lp_i / _pt_lp_i.sum()))
            _Ept_lp_i = -np.log(np.nan_to_num(_pt_lp_i))

            shape_pt_i_pad = (
                (0, K_max - _speed_i.shape[0]),
                (0, K_max - _speed_i_1.shape[0]),
            )
            _Ept_lp_i = np.pad(
                _Ept_lp_i, shape_pt_i_pad, "constant", constant_values=-1
            )
            _Ept_lp.append(np.nan_to_num(_Ept_lp_i))

    return np.nan_to_num(np.stack(_Ept_lp, axis=0))


def getlpmuE(lp_v_mu):
    lpmuE = -(lp_v_mu / np.sum(lp_v_mu) * np.log(lp_v_mu / np.sum(lp_v_mu)))
    # lpmuE = - np.log(lp_v_mu / np.sum(lp_v_mu))
    lpmuE_sum = lpmuE.sum()
    return lpmuE, lpmuE_sum


def getST_rec(STsegtag, ST_seg, seg_lp_argmax_k):
    ST_rec = list()
    _ST_seg = ST_seg[ST_seg["tlSeg"] == STsegtag].copy()
    _ST_seg = _ST_seg[(_ST_seg["odpairs"].str.contains("ignore") == True)==False]
    if len(seg_lp_argmax_k)!=len(_ST_seg):
        return "error"
    else:
	    _ST_seg["seg_lp_argmax_k"] = seg_lp_argmax_k
	    ST_rec = _ST_seg.parallel_apply(
	        lambda seg: seg["odpaths"][seg["seg_lp_argmax_k"]], axis=1
	    ).tolist()
	
	    return ST_rec


def ignorejudge(i, o_d_k):
    tlsegs = ST_seg.groupby("tlSeg").get_group("0_0")
    tlsegs.reset_index(inplace=True)

    # print(i,tlsegs.loc[i]["odpaths"][o_d_k[0]],tlsegs.loc[i+1]["odpaths"][o_d_k[1]])
    # return True

    if (
        tlsegs.loc[i]["odpaths"][o_d_k[0]][-1]
        == tlsegs.loc[i + 1]["odpaths"][o_d_k[1]][0]
    ):
        # print('here')
        return True
    else:
        return False


def socre(Epe_lp, Ept_lp):
    """Decode the highest scoring sequence of tags outside of TensorFlow.

    This should only be used at test time.

    Args:
        score: A [seq_len, num_tags] matrix of unary potentials.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.

    Returns:
        viterbi: A [seq_len] list of integers containing the highest scoring tag
                indices.
        viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(Epe_lp)
    # 用于存储最优路径索引的数组
    backpointers = np.zeros_like(Epe_lp, dtype=np.int32)

    # 第一个时刻的累计分数
    trellis[:, 0] = Epe_lp[:, 0]

    viterbi_ = list()

    for t in range(1, Epe_lp.shape[1]):

        # 各个状态截止到上个时刻的累计分数 + 转移分数
        v = trellis[:, t - 1] + Ept_lp[t]
        # max（各个状态截止到上个时刻的累计分数 + 转移分数）+ 选择当前状态的分数
        trellis[:, t] = Epe_lp[:, t] + np.max(v, 1)
        # 记录累计分数最大的索引
        backpointers[:, t] = np.argmax(v, 1)
        viterbi_.append(backpointers[:, t][np.argmax(trellis[:, t])])

    viterbi_.append(np.argmax(trellis[:, -1]))
    # 最优路径的结果
    viterbi = [np.argmax(trellis[:, -1])]
    # print(viterbi[-1],np.argmax(trellis[:,-1]))
    for bp in reversed(backpointers.T[1:]):
        # print(viterbi[-1],bp)
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[:, -1])

    return viterbi, viterbi_score


def _socre(Epe_lp, Ept_lp):
    """Decode the highest scoring sequence of tags outside of TensorFlow.

    This should only be used at test time.

    Args:
        score: A [seq_len, num_tags] matrix of unary potentials.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.

    Returns:
        viterbi: A [seq_len] list of integers containing the highest scoring tag
                indices.
        viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(Epe_lp)
    # 用于存储最优路径索引的数组
    backpointers = np.zeros_like(Epe_lp, dtype=np.int32)

    # 第一个时刻的累计分数
    trellis[:, 0] = Epe_lp[:, 0]

    viterbi_ = list()

    for t in range(1, Epe_lp.shape[1]):

        if t - 1 < 2:
            # 各个状态截止到上个时刻的累计分数 + 转移分数
            v = trellis[:, t - 1] + Ept_lp[t]

            # max（各个状态截止到上个时刻的累计分数 + 转移分数）+ 选择当前状态的分数
            trellis[:, t] = Epe_lp[:, t] + np.max(v, 1)

            # 记录累计分数最大的索引
            backpointers[:, t] = np.argmax(v, 1)
            viterbi_.append(backpointers[:, t][np.argmax(trellis[:, t])])
            continue

        # temp = trellis[t].copy()
        v_ = trellis[:, t - 1] + Ept_lp[t]
        trellis_ = Epe_lp[:, t] + np.max(v_, 1)
        backpointers_ = np.argmax(v_, 1)
        index_del = np.empty((0))
        tt = backpointers_[np.argmax(trellis_)]
        o_d_k = [viterbi_[-1], tt]
        flag = False
        while True:
            flag = False
            # print(t-2,o_d_k)
            if trellis_.sum() == 0:
                # print('hhhhhhddddddddddddddddd')
                for p in range(len(backpointers_)):
                    o_d_k[1] = p
                    # print("ppp",t-2,o_d_k)
                    if ignorejudge(t - 2, o_d_k):
                        # print(t-2,o_d_k)
                        flag = True
                        break
                if flag:
                    break
            else:
                if ignorejudge(t - 2, o_d_k):
                    break
                else:
                    # temp_max = np.max(temp)

                    index_max = np.where(trellis_ == np.max(trellis_))[0]

                    index_del = np.append(index_del, index_max).astype(int)
                    # print(index_del)
                    trellis_[index_del] = 0
                    # print(np.argmax(trellis[t]),np.max(trellis[t]))
                    o_d_k[1] = backpointers_[np.argmax(trellis_)]
                    # print(t-3,o_d_k,backpointers[t])
        tt = o_d_k[1]
        trellis[:, t] = trellis_
        backpointers[:, t] = np.argmax(v_, 1)
        viterbi_.append(tt)

    viterbi_.append(np.argmax(trellis[:, -1]))
    # 最优路径的结果
    viterbi = [np.argmax(trellis[:, -1])]
    # print(viterbi[-1],np.argmax(trellis[:,-1]))
    for bp in reversed(backpointers.T[1:]):
        # print(viterbi[-1],bp)
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[:, -1])

    return viterbi, viterbi_score


def recon_func(
    pe_lp_arg,
    pt_lp_arg,
    segE_var,
    segE_arg,
    st_eulid_dur,
    lp_path_basicvar,
    K_max,
    nbest=1,
):

    _lpmuE_sum_list = list()
    _score_list = list()
    _seg_lp_argmax_k_list = list()
    _lp_v_mu_list = list()

    Epe_lp = pe_lp_i(lp_path_basicvar, st_eulid_dur, pe_lp_arg, K_max)

    fp.info(f"Epe_lp finish")

    segE = getsegE(segE_var, segE_arg)

    (
        init_lp_argmax_k,
        _lp_v_mu,
        _lp_v_sigma,
        _ST_v_mu,
        _ST_v_sigma,
    ) = getST_v_mu_sigma_init(Epe_lp, lp_path_basicvar[:, :, 1].T)
    
    lp_path_speed, lp_v_mu, lp_v_sigma, ST_v_mu, ST_v_sigma = update_lp_v_mu_sigma_ab(
        segE,
        lp_path_basicvar[:, :, 1].T,
        _lp_v_mu,
        _lp_v_sigma,
        _ST_v_mu,
        _ST_v_sigma,
        pt_lp_arg,
    )
    init_lp_v_mu = lp_v_mu

    init_lpmuE_sum = getlpmuE(lp_v_mu)[1]

    itr = 0
    _lpmuE_set_ = 0
    while True:
        pt_lp_var = {
            "segE": segE,
            "K_max": K_max,
            "lp_path_speed": lp_path_speed,
            "lp_v_sigma": lp_v_sigma,
            "ST_v_sigma": ST_v_sigma,
        }

        Ept_lp = pt_lp_i_1step(pt_lp_var)
        # Ept_lp = pt_lp_i_afb(pt_lp_var, pt_lp_arg)

        fp.info(f"Ept_lp finish using pt_lp_i_afb")

        """
        # seg_lp_argmax_k, score = socre(Epe_lp, Ept_lp)
        path_score, decode_idx = viterbi_decode_nbest(
            Epe_lp.T.reshape([1, Epe_lp.shape[1], Epe_lp.shape[0]]), Ept_lp, 1
        )

        seg_lp_argmax_k = decode_idx.transpose([0, 2, 1])[0][0].tolist()
        path_score = path_score[0][0].tolist()

        _lp_v_mu, _ST_v_mu, _ST_v_sigma = getST_v_mu_sigma(
            lp_path_speed, seg_lp_argmax_k
        )

        (
            lp_path_speed,
            lp_v_mu,
            lp_v_sigma,
            ST_v_mu,
            ST_v_sigma,
        ) = update_lp_v_mu_sigma_ab(
            segE, lp_path_speed, _lp_v_mu, _lp_v_sigma, _ST_v_mu, _ST_v_sigma, pt_lp_arg
        )
        lpmuE_sum = getlpmuE(lp_v_mu)[1]

        _lpmuE_sum_list.append(lpmuE_sum)
        _lp_v_mu_list.append(lp_v_mu)

        """
        path_score, decode_idx = viterbi_decode_nbest(
            Epe_lp.T.reshape([1, Epe_lp.shape[1], Epe_lp.shape[0]]), Ept_lp, nbest
        )
        seg_lp_argmax_k_nbest = decode_idx.transpose([0, 2, 1])[0].tolist()
        path_score = path_score[0].tolist()

        fp.info(f"seg_lp_argmax_k finish using viterbi_decode")
        for seg_lp_argmax_k in seg_lp_argmax_k_nbest:
            _lp_v_mu, _lp_v_sigma,_ST_v_mu, _ST_v_sigma = getST_v_mu_sigma(
                lp_path_speed, seg_lp_argmax_k
            )

            (
                lp_path_speed,
                lp_v_mu,
                lp_v_sigma,
                ST_v_mu,
                ST_v_sigma,
            ) = update_lp_v_mu_sigma_ab(
                segE,
                lp_path_speed,
                _lp_v_mu,
                _lp_v_sigma,
                _ST_v_mu,
                _ST_v_sigma,
                pt_lp_arg,
            )
            lpmuE_sum = getlpmuE(lp_v_mu)[1]

            _lpmuE_sum_list.append(lpmuE_sum)
            _lp_v_mu_list.append(lp_v_mu)

        itr += 1
        if itr > 10:
            _lpmuE_sum_list.pop()
            _lpmuE_sum_list.pop()
            _lpmuE_sum_list.pop()
            _lp_v_mu_list.pop()
            _lp_v_mu_list.pop()
            _lp_v_mu_list.pop()
            break
        if len(set(_lpmuE_sum_list)) == _lpmuE_set_:
            _lpmuE_sum_list.pop()
            _lpmuE_sum_list.pop()
            _lpmuE_sum_list.pop()
            _lp_v_mu_list.pop()
            _lp_v_mu_list.pop()
            _lp_v_mu_list.pop()
            break
        else:
            _lpmuE_set_ = len(set(_lpmuE_sum_list))
            _seg_lp_argmax_k_list.append(seg_lp_argmax_k_nbest)  # seg_lp_argmax_k_nbest
            _score_list.append(path_score)  # path_score
            # _lp_v_mu_list.append(lp_v_mu)

        fp.info(f"recon_func itrs={itr}")
        # break
    fp.info(f"recon_func finish")

    return (
        _seg_lp_argmax_k_list,
        _lpmuE_sum_list,
        _score_list,
        _lp_v_mu_list,
        init_lp_argmax_k,
        init_lpmuE_sum,
        init_lp_v_mu,
        Epe_lp,
        Ept_lp,
        lp_path_speed,
        itr - 1,
    )


def viterbi_decode_nbest(emission_score, transition_score, nbest=3):
    """
    input:
        emission_score: (batch, seq_len, tag_size)
        mask: (batch, seq_len)
        transition_score: (tag_size,tag_size)
    output:
        decode_idx: (batch, seq_len, nbest) decoded sequence
        path_score: (batch, nbest)
    """
    START_TAG = 0  # START_TAG的标签集id为倒数第2个
    STOP_TAG = -1  # STOP_TAG的标签集id为最后一个
    batch_size = emission_score.shape[0]
    seq_len = emission_score.shape[1]  # batch中最长的句子长度
    tag_size = emission_score.shape[2]  # 标签集大小，START_TAG,END_TAG
    mask = np.ones([seq_len], dtype=int).reshape([batch_size, seq_len])
    length_mask = np.sum(mask, axis=1).reshape([batch_size, 1])  # (batch_size,1)的列向量
    ins_num = seq_len * batch_size

    mask = mask.transpose((1, 0))  # mask转置成(seq_len, batch_size)形状
    emission_score = emission_score.transpose([1, 0, 2]).reshape([ins_num, 1, tag_size])

    scores = emission_score + transition_score  # .reshape([1, tag_size, tag_size])
    scores = scores.reshape([seq_len, batch_size, tag_size, tag_size])

    seq_iter = enumerate(scores)

    back_points = list()
    nbest_scores_t_history = list()

    t, scores_t0 = next(seq_iter)  # scores_t0.shape = (batch_size,tag_size,tag_size)

    nbest_scores_t = scores_t0[:, START_TAG, :]  # (batch_size,tag_size)

    nbest_scores_t_history.append(
        np.tile(nbest_scores_t.reshape([batch_size, tag_size, 1]), [1, 1, nbest])
    )

    for t, scores_t in seq_iter:
        if t == 1:
            scores_t = scores_t.reshape(
                [batch_size, tag_size, tag_size]
            ) + nbest_scores_t.reshape([batch_size, tag_size, 1])
        else:
            scores_t = np.tile(
                scores_t.reshape([batch_size, tag_size, 1, tag_size]), [1, 1, nbest, 1]
            ) + np.tile(
                nbest_scores_t.reshape([batch_size, tag_size, nbest, 1]),
                [1, 1, 1, tag_size],
            )
            scores_t = scores_t.reshape([batch_size, tag_size * nbest, tag_size])

        cur_bp = np.argsort(scores_t, axis=1)[:, -nbest:][:, ::-1, :]
        # cur_bp为(batch_size,nbest,tag_size)形状的index，每个值的意义为，忽略第0维batch_size时，每一列前n个最大值的行号
        # argsort会做全排序，此处可以先argnbest_scores_t做局部排序，然后对前n个序列再做全排序，提高效率
        nbest_scores_t = scores_t[
            np.tile(
                np.arange(0, batch_size).reshape(batch_size, 1, 1), [1, nbest, tag_size]
            ),
            cur_bp,
            np.tile(
                np.arange(0, tag_size).reshape([1, 1, tag_size]), [batch_size, nbest, 1]
            ),
        ]
        # nbest_scores_t为在scores_t中每一行的topk取出的值(batch_size,nbest,tag_size)
        if t == 1:
            cur_bp = cur_bp * nbest
        nbest_scores_t = nbest_scores_t.transpose([0, 2, 1])
        # 形状为(batch_size,tag_size,nbest)，转置是因为在(tag_size,tag_size)的转移矩阵中，
        # 行号为上一时刻的tag编号，列号为当前时刻的tag编号，所以当前的n个最优序列需要转置后广播作为下一个时刻的行号，即出发tag
        cur_bp = cur_bp.transpose([0, 2, 1])  # 形状为(batch_size,tag_size,nbest)
        nbest_scores_t_history.append(nbest_scores_t)
        cur_bp = np.multiply(
            cur_bp, np.tile(mask[t].reshape([batch_size, 1, 1]), [1, tag_size, nbest])
        )
        back_points.append(cur_bp)
    nbest_scores_t_history = (
        np.concatenate(nbest_scores_t_history, axis=0)
        .reshape([seq_len, batch_size, tag_size, nbest])
        .transpose([1, 0, 2, 3])
    )
    ## (batch_size, seq_len, tag_size, nbest)
    last_position = (
        np.tile(length_mask.reshape([batch_size, 1, 1, 1]), [1, 1, tag_size, nbest]) - 1
    )
    last_nbest_scores = nbest_scores_t_history[
        np.tile(
            np.arange(batch_size).reshape([batch_size, 1, 1, 1]),
            [1, 1, tag_size, nbest],
        ),
        last_position,
        np.tile(
            np.arange(tag_size).reshape([1, 1, tag_size, 1]), [batch_size, 1, 1, nbest]
        ),
        np.tile(
            np.arange(nbest).reshape([1, 1, 1, nbest]), [batch_size, 1, tag_size, 1]
        ),
    ]
    # 形状为(batch_size,1,tag_size,nbest)
    # 获取每个batch中example最后有效时间步的nbest_scores_t矩阵

    # 计算最后一个时间步转移到END_TAG的过程
    last_nbest_scores = last_nbest_scores.reshape([batch_size, tag_size, nbest, 1])
    last_values = np.tile(last_nbest_scores, [1, 1, 1, tag_size]) + np.tile(
        transition_score[seq_len - 1].reshape(1, tag_size, 1, tag_size),
        [batch_size, 1, nbest, 1],
    )
    last_values = last_values.reshape([batch_size, tag_size * nbest, tag_size])
    # last_values = np.tile(last_nbest_scores, [1, 1, 1, tag_size])
    # last_values = last_values.reshape([batch_size, tag_size * nbest, tag_size])
    end_bp = np.argsort(last_values, axis=1)[:, -nbest:][:, ::-1, :]
    end_nbest_scores = last_values[
        np.tile(
            np.arange(0, batch_size).reshape(batch_size, 1, 1), [1, nbest, tag_size]
        ),
        end_bp,
        np.tile(
            np.arange(0, tag_size).reshape([1, 1, tag_size]), [batch_size, nbest, 1]
        ),
    ]
    # end_nbest_scores为在last_values中每一列的topk取出的值(batch_size,nbest,tag_size)
    end_bp = end_bp.transpose([0, 2, 1])
    # 形状为(batch_size,tag_size,nbest)
    pad_zero = np.zeros([batch_size, tag_size, nbest], dtype=np.int32)
    back_points.append(pad_zero)
    back_points = np.concatenate(back_points, axis=0).reshape(
        [seq_len, batch_size, tag_size, nbest]
    )
    # (seq_len,batch_size,tag_size,nbest)
    last_pointer = end_bp[:, STOP_TAG, :]
    # (batch_size, nbest)
    insert_last = np.tile(
        last_pointer.reshape([batch_size, 1, 1, nbest]), [1, 1, tag_size, 1]
    )
    # (batch_size,1,tag_size,nbest)
    back_points = back_points.transpose([1, 0, 2, 3])
    # (batch_size,seq_len,tag_size,nbest)

    back_points[
        np.tile(
            np.arange(0, batch_size).reshape([batch_size, 1, 1, 1]),
            [1, 1, tag_size, nbest],
        ),
        last_position,
        np.tile(
            np.arange(0, tag_size).reshape([1, 1, tag_size, 1]),
            [batch_size, 1, 1, nbest],
        ),
        np.tile(
            np.arange(0, nbest).reshape([1, 1, 1, nbest]), [batch_size, 1, tag_size, 1]
        ),
    ] = insert_last
    # 把back_points中每个example对应的最后一个时间步的(tag_size,nbest)的全零矩阵改成insert_last中对应example的矩阵
    back_points = back_points.transpose([1, 0, 2, 3])
    # (seq_len, batch_size, tag_size, nbest)
    decode_idx = np.zeros([seq_len, batch_size, nbest], dtype=np.int32)
    decode_idx[-1] = (last_pointer / nbest).astype(np.int32)
    mask_1 = np.array(
        [([1] * n + [0] * (seq_len - n)) for n in length_mask.reshape([batch_size]) - 1]
    ).transpose([1, 0])
    # (seq_len,batch_size)
    pointer = last_pointer.copy()
    for t in range(len(back_points) - 2, -1, -1):
        new_pointer = back_points[t].reshape([batch_size, tag_size * nbest])[
            np.tile(np.arange(0, batch_size).reshape(batch_size, 1), [1, nbest]),
            pointer.reshape([batch_size, nbest]).astype(np.int32),
        ]
        new_pointer = new_pointer * mask_1[t].reshape(
            [batch_size, 1]
        ) + last_pointer.reshape([batch_size, nbest]) * (1 - mask_1[t]).reshape(
            [batch_size, 1]
        )
        # 每个example最后一个tag直接使用last_pointer的值，往前再开始顺藤摸瓜
        decode_idx[t] = (new_pointer / nbest).astype(np.int32)
        # # use new pointer to remember the last end nbest ids for non longest
        pointer = new_pointer
    decode_idx = decode_idx.transpose([1, 0, 2])
    # (batch_size,seq_len,nbest)
    scores = end_nbest_scores[:, :, STOP_TAG]

    def softmax(x, axis=1):
        max_x = np.max(x, axis=axis, keepdims=True)
        minus_x = x - max_x
        return np.exp(minus_x) / np.sum(np.exp(minus_x), axis=axis, keepdims=True)

    path_score = softmax(scores)

    return path_score, decode_idx