import copy
import math
import warnings

import geopandas as gpd
import geopy.distance
import numpy as np
import pandas as pd
import transbigdata as tbd
from geopy.distance import geodesic
from pandarallel import pandarallel
from shapely import unary_union, wkt
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from .. import basicfunc, creategraph, logs, stats, util_coord, utils_graph


def flatten(li):
    return sum(([x] if not isinstance(x, list) else flatten(x) for x in li), [])


def _getbestK_(segseg, traj_points, network_grid_cl_edges_info, grid_bound, buffer=20):
    if "ignore" in segseg["odpaths"]:
        return "ignore", "ignore"
    else:
        traj_points_seg_i = traj_points[
            traj_points["timestamp"].between(
                segseg["time_o"], segseg["time_d"], inclusive="both"
            )
        ]

        if len(traj_points_seg_i) == 1:
            temp = pd.concat(
                [
                    pd.DataFrame(
                        [
                            {
                                "lat": segseg["lat_o"],
                                "lng": segseg["lng_o"],
                                "entity_id": traj_points_seg_i["entity_id"].values[0],
                            }
                        ]
                    ),
                    traj_points_seg_i,
                ]
            )
            temp = pd.concat(
                [
                    temp,
                    pd.DataFrame(
                        [
                            {
                                "lat": segseg["lat_d"],
                                "lng": segseg["lng_d"],
                                "entity_id": traj_points_seg_i["entity_id"].values[0],
                            }
                        ]
                    ),
                ]
            )
            traj = tbd.points_to_traj(temp, col=["lng", "lat", "entity_id"]).set_crs(
                4326
            )
        elif len(traj_points_seg_i) == 0:
            temp = pd.DataFrame(
                {
                    "lat": [segseg["lat_o"], segseg["lat_d"]],
                    "lng": [segseg["lng_o"], segseg["lng_d"]],
                    "entity_id": traj_points["entity_id"].values[0],
                }
            )
            traj = tbd.points_to_traj(temp, col=["lng", "lat", "entity_id"]).set_crs(
                4326
            )
        else:
            traj = tbd.points_to_traj(
                traj_points_seg_i, col=["lng", "lat", "entity_id"]
            ).set_crs(4326)

        path_length = basicfunc.getlinestringlen(traj["geometry"].values[0])

        pathgrids = list()
        pathgrids_length = list()
        for k, _shortest_path in enumerate(segseg["odpaths"]):
            _pathgrids_ = network_grid_cl_edges_info.loc[
                list(zip(_shortest_path[:-1], _shortest_path[1:]))
            ]
            _pathgrids = (
                grid_bound.set_index("grid")
                .loc[list(set(flatten(_pathgrids_["edge_grid"].tolist())))]
                .copy()
            )

            pathgrids.append(
                gpd.GeoDataFrame(
                    [
                        {
                            "geometry": unary_union(_pathgrids["geometry"]),
                            "_length_": _pathgrids_["_length_"].sum(),
                            "k": k,
                        }
                    ],
                    crs="4326",
                )
            )
            pathgrids_length.append(_pathgrids_["_length_"].sum())

        gpd_pathgrids = gpd.GeoDataFrame(pd.concat(pathgrids))
        gpd_pathgrids["geometry"] = (
            gpd_pathgrids.to_crs("epsg:3395").buffer(buffer).to_crs("epsg:4326")
        )
        gpd_pathgrids["rec_inter"] = gpd_pathgrids.intersection(traj["geometry"])
        gpd_pathgrids["rec_length"] = gpd_pathgrids["rec_inter"].apply(
            lambda _rec_inter_: sum(
                [basicfunc.getlinestringlen(_line) for _line in _rec_inter_.geoms]
            )
            if "MultiLineString" in str(type(_rec_inter_))
            else basicfunc.getlinestringlen(_rec_inter_)
        )
        if gpd_pathgrids["rec_length"].sum() == 0:
            return "ignore_badrec", "ignore_badrec"
        gpd_pathgrids["rec_area"] = gpd_pathgrids.to_crs(3395).area
        gpd_pathgrids.set_index("k", inplace=True)

        gpd_pathgrids["rec_ratio"] = (
            1 - abs(path_length - gpd_pathgrids["rec_length"]) / path_length
        )
        if gpd_pathgrids["rec_ratio"].mean() < 0.7:
            return "ignore_badrec", "ignore_badrec"
        # _best_k = gpd_pathgrids[
        #    gpd_pathgrids["_length_"]
        #    == gpd_pathgrids[
        #        gpd_pathgrids["rec_ratio"] == gpd_pathgrids["rec_ratio"].max()
        #    ]["_length_"].min()
        # ].index.values[0]
        _best_k = gpd_pathgrids[
            gpd_pathgrids["_length_"]
            == gpd_pathgrids[
                gpd_pathgrids["rec_area"]
                == gpd_pathgrids[
                    gpd_pathgrids["rec_ratio"] == gpd_pathgrids["rec_ratio"].max()
                ]["rec_area"].min()
            ]["_length_"].min()
        ].index.values[0]
        _rec_ratio = gpd_pathgrids.loc[_best_k]["rec_ratio"]
        return _best_k, _rec_ratio


def getbestK(
    _STtr,
    _ST_seg,
    _network_grid_cl_edges_info,
    _grid_bound,
    _speedlimit=80,
    _dislimit=500,
    _buffer=20,
):
    ST_seg = _ST_seg.copy()
    id_object = _STtr.iloc[0]["id_object"]
    id_path = _STtr.iloc[0]["id_pathset"]
    _traj_points = pd.read_csv(
        f"gpsRawData/{id_object}/Trajectory/{id_path}.plt",
        header=5,
        names=["lat", "lng", "zero", "alt", "days", "date", "time"],
        index_col=False,
    )
    _traj_points["timestamp"] = _traj_points["date"] + " " + _traj_points["time"]
    _traj_points["timestamp"] = pd.to_datetime(_traj_points["timestamp"])
    _traj_points["entity_id"] = id_object
    traj_points = tbd.clean_drift(
        _traj_points,
        col=["entity_id", "time", "lng", "lat"],
        speedlimit=_speedlimit,
        dislimit=_dislimit,
    )

    ST_seg[["best_k", "rec_ratio"]] = ST_seg.parallel_apply(
        lambda segseg: _getbestK_(
            segseg, traj_points, _network_grid_cl_edges_info, _grid_bound, _buffer
        ),
        axis=1,
        result_type="expand",
    )
    ST_seg["odpaths"] = ST_seg.parallel_apply(
        lambda segseg: "ignore"
        if "ignore" in str(segseg["best_k"])
        else segseg["odpaths"],
        axis=1,
    )
    return ST_seg


def get_ob_var(lp_path_basicvar, st_eulid_dur, K_max):
    _ob_var_list = list()
    for i in range(0, len(lp_path_basicvar)):
        i_len_eulid = st_eulid_dur[i, :][0]
        i_duration = st_eulid_dur[i, :][1]

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

        _var_lp_speed = i_lp_len / i_duration
        _var_lp_len = 1 - (
            abs(i_lp_len - i_len_eulid) - abs(i_lp_len - i_len_eulid).min()
        ) / (abs(i_lp_len - i_len_eulid).max() - abs(i_lp_len - i_len_eulid).min())

        _var_lp_dir_head = 1 - (i_lp_dir_head - i_lp_dir_head.min()) / (
            i_lp_dir_head.max() - i_lp_dir_head.min()
        )

        _var_lp_dir_trun = 1 - (i_lp_dir_turn - i_lp_dir_turn.min()) / (
            i_lp_dir_turn.max() - i_lp_dir_turn.min()
        )

        _var_lp_det_count = 1 - (i_lp_det_count - i_lp_det_count.min()) / (
            i_lp_det_count.max() - i_lp_det_count.min()
        )

        _var_lp_det_dist = 1 - (i_lp_det_dist - i_lp_det_dist.min()) / (
            i_lp_det_dist.max() - i_lp_det_dist.min()
        )

        _var_lp_det_len = 1 - (i_lp_det_len - i_lp_det_len.min()) / (
            i_lp_det_len.max() - i_lp_det_len.min()
        )

        var_lp_speed = np.pad(
            _var_lp_speed,
            (0, K_max - len(_var_lp_speed)),
            "constant",
            constant_values=-1,
        )
        var_lp_len = np.pad(
            _var_lp_len, (0, K_max - len(_var_lp_len)), "constant", constant_values=-1
        )
        var_lp_dir_head = np.pad(
            _var_lp_dir_head,
            (0, K_max - len(_var_lp_dir_head)),
            "constant",
            constant_values=-1,
        )
        var_lp_dir_trun = np.pad(
            _var_lp_dir_trun,
            (0, K_max - len(_var_lp_dir_trun)),
            "constant",
            constant_values=-1,
        )
        var_lp_det_count = np.pad(
            _var_lp_det_count,
            (0, K_max - len(_var_lp_det_count)),
            "constant",
            constant_values=-1,
        )
        var_lp_det_dist = np.pad(
            _var_lp_det_dist,
            (0, K_max - len(_var_lp_det_dist)),
            "constant",
            constant_values=-1,
        )
        var_lp_det_len = np.pad(
            _var_lp_det_len,
            (0, K_max - len(_var_lp_det_len)),
            "constant",
            constant_values=-1,
        )

        _ob_var_list.append(
            np.stack(
                [
                    var_lp_speed,
                    var_lp_len,
                    var_lp_dir_head,
                    var_lp_dir_trun,
                    var_lp_det_count,
                    var_lp_det_dist,
                    var_lp_det_len,
                ],
                axis=1,
            )
        )

    _ob_var = np.nan_to_num(np.stack(_ob_var_list, axis=1))
    return _ob_var


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

    return np.nan_to_num(segE)


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