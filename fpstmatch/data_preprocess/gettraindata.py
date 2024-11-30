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
from shapely import ops, unary_union, wkt
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from .. import basicfunc, creategraph, logs, stats, util_coord, utils_graph


def getgridparams(bound, accuracy):
    lon1, lat1, lon2, lat2 = bound
    if (lon1 > lon2) | (lat1 > lat2) | (abs(lat1) > 90) | (abs(lon1) > 180) | (
            abs(lat2) > 90) | (abs(lon2) > 180):
        raise Exception(  # pragma: no cover
            'Bounds error. The input bounds should be in the order '
            'of [lon1,lat1,lon2,lat2]. (lon1,lat1) is the lower left '
            'corner and (lon2,lat2) is the upper right corner.'
        )
    latStart = min(lat1, lat2)
    lonStart = min(lon1, lon2)
    deltaLon = accuracy * 360 / \
        (2 * math.pi * 6371004 * math.cos((lat1 + lat2) * math.pi / 360))
    deltaLat = accuracy * 360 / (2 * math.pi * 6371004)
    _params_rect = {'slon': lonStart,
                    'slat': latStart,
                    'deltalon': deltaLon,
                    'deltalat': deltaLat,
                    'theta': 0,
                    'method': "rect",
                    'gridsize': accuracy}
    _params_hexa = {'slon': lonStart,
                    'slat': latStart,
                    'deltalon': deltaLon,
                    'deltalat': deltaLat,
                    'theta': 0,
                    'method': "hexa",
                    'gridsize': accuracy}
    return _params_rect, _params_hexa


def getdetectorinfo(bound, det_db_name, conn, params_grid):
    bound_min = [bound[0], bound[1]]
    bound_max = [bound[2], bound[3]]

    bound_polygon_str = f"POLYGON(({bound_min[0]} {bound_min[1]},{bound_min[0]} {bound_max[1]},{bound_max[0]} {bound_max[1]},{bound_max[0]} {bound_min[1]},{bound_min[0]} {bound_min[1]}))"
    st_geomfromtext_str = f"ST_GeomFromText('{bound_polygon_str}',4326)"
    st_contains_str = f"ST_Intersects({st_geomfromtext_str},geometry)"

    sql_slt_det = f"SELECT * FROM {det_db_name} where {st_contains_str}"

    detectors_inarea = gpd.GeoDataFrame.from_postgis(
        sql_slt_det, conn, "geometry", crs="4326").set_index("name")
    detectors_inarea["loncol_1"], detectors_inarea["loncol_2"], detectors_inarea["loncol_3"] = tbd.GPS_to_grid(
        detectors_inarea["lng"], detectors_inarea["lat"], params_grid)
    detectors_inarea["grid"] = detectors_inarea["loncol_1"].astype(
        str) + "_" + detectors_inarea["loncol_2"].astype(str) + "_" + detectors_inarea["loncol_3"].astype(str)
    detectors_inarea["point"] = gpd.points_from_xy(
        detectors_inarea.lng, detectors_inarea.lat)
    detectors_inarea["grids"] = detectors_inarea.apply(
        lambda detector: getdetetorgrids(detector, params_grid), axis=1)

    return detectors_inarea


def getdetetorgrids(_detector, params_grid):
    bounds = _detector["geometry"].bounds
    deltaLon = params_grid["deltalon"]
    deltaLat = params_grid["deltalat"]
    tmppoints = pd.DataFrame(
        np.array(
            np.meshgrid(
                np.arange(bounds[0],
                          bounds[2],
                          deltaLon/3),
                np.arange(bounds[1],
                          bounds[3],
                          deltaLat/3))
        ).reshape(2, -1).T, columns=['lon', 'lat'])
    tmppoints['loncol_1'],\
        tmppoints['loncol_2'],\
        tmppoints['loncol_3'] = tbd.GPS_to_grid(
            tmppoints['lon'], tmppoints['lat'], params_grid)
    tmppoints = tmppoints[['loncol_1',
                           'loncol_2', 'loncol_3']].drop_duplicates()
    tmppoints["grid"] = tmppoints["loncol_1"].astype(
        str) + "_" + tmppoints["loncol_2"].astype(str) + "_" + tmppoints["loncol_3"].astype(str)

    return tmppoints["grid"].tolist()


def getSTseg(STtr, delta_T, delta_L):
    ST_seg = pd.DataFrame()

    #ST_seg["id"] = STtr["index"]
    ST_seg["object"] = STtr["id_object"]
    ST_seg["pathset"] = STtr["id_pathset"]

    ST_seg["det_name_o"] = STtr["name"].shift()
    ST_seg["det_name_d"] = STtr["name"]

    ST_seg["det_area_o"] = STtr["geometry"].shift()
    ST_seg["det_area_d"] = STtr["geometry"]

    ST_seg["lat_o"] = STtr["lat"].shift()
    ST_seg["lng_o"] = STtr["lng"].shift()
    ST_seg["lat_d"] = STtr["lat"]
    ST_seg["lng_d"] = STtr["lng"]

    ST_seg["time_o"] = STtr["time"].shift()
    ST_seg["time_d"] = STtr["time"]

    ST_seg["tSeg"] = (
        ((STtr["time"] - STtr["time"].shift()) > delta_T).astype(int).cumsum()
    )
    ST_seg["lSeg"] = (
        (
            tbd.getdistance(
                STtr["lng"],
                STtr["lat"],
                STtr["lng"].shift(),
                STtr["lat"].shift(),
            )
            > delta_L
        )
        .astype(int)
        .cumsum()
    )

    for ts, subtr in ST_seg.groupby("tSeg"):
        ST_seg.loc[ST_seg[ST_seg["tSeg"].isin([ts])].index, "lSeg"] = (
            subtr["lSeg"] - subtr["lSeg"].min()
        )

    ST_seg["tlSeg"] = ST_seg["tSeg"].astype(
        str) + "_" + ST_seg["lSeg"].astype(str)

    ST_seg["dist_nsp"] = tbd.getdistance(
        ST_seg["lng_o"],
        ST_seg["lat_o"],
        ST_seg["lng_d"],
        ST_seg["lat_d"],
    )

    ST_seg["dur_nsp"] = ST_seg["time_d"] - ST_seg["time_o"]

    for ts, subtr in ST_seg.groupby("tlSeg"):
        ST_seg.drop(subtr.head(1).index, inplace=True)
    ST_seg.reset_index(inplace=True)
    return ST_seg


def traj_to_grids(traj, params):
    shape = traj
    bounds = shape.unary_union.bounds

    lon1, lat1, lon2, lat2 = bounds
    if (lon1 > lon2) | (lat1 > lat2) | (abs(lat1) > 90) | (abs(lon1) > 180) | (
            abs(lat2) > 90) | (abs(lon2) > 180):
        raise Exception(  # pragma: no cover
            'Bounds error. The input bounds should be in the order '
            'of [lon1,lat1,lon2,lat2]. (lon1,lat1) is the lower left '
            'corner and (lon2,lat2) is the upper right corner.'
        )

    deltaLon = params["gridsize"] * 360 / \
        (2 * math.pi * 6371004 * math.cos((lat1 + lat2) * math.pi / 360))
    deltaLat = params["gridsize"] * 360 / (2 * math.pi * 6371004)

    method = params['method']

    if (method == 'tri') | (method == 'hexa'):
        longapsize = np.arange(bounds[0], bounds[2], deltaLon/3).size
        latgapsize = np.arange(bounds[1], bounds[3], deltaLat/3).size
        if longapsize == 0 and latgapsize != 0:
            tmppoints = pd.DataFrame(
                np.array(
                    np.meshgrid(
                        np.array([bounds[0]]*latgapsize),
                        np.arange(bounds[1],
                                  bounds[3],
                                  deltaLat/3))
                ).reshape(2, -1).T, columns=['lon', 'lat'])
        elif latgapsize == 0 and longapsize != 0:
            tmppoints = pd.DataFrame(
                np.array(
                    np.meshgrid(
                        np.arange(bounds[0],
                                  bounds[2],
                                  deltaLon/3),
                        np.array([bounds[1]]*longapsize))
                ).reshape(2, -1).T, columns=['lon', 'lat'])
        else:
            tmppoints = pd.DataFrame(
                np.array(
                    np.meshgrid(
                        np.arange(bounds[0],
                                  bounds[2],
                                  deltaLon/3),
                        np.arange(bounds[1],
                                  bounds[3],
                                  deltaLat/3))
                ).reshape(2, -1).T, columns=['lon', 'lat'])
        tmppoints['loncol_1'],\
            tmppoints['loncol_2'],\
            tmppoints['loncol_3'] = tbd.GPS_to_grid(
            tmppoints['lon'], tmppoints['lat'], params)
        tmppoints = tmppoints[['loncol_1',
                               'loncol_2', 'loncol_3']].drop_duplicates()
        tmppoints['geometry'] = tbd.grid_to_polygon(
            [tmppoints['loncol_1'],
                tmppoints['loncol_2'],
                tmppoints['loncol_3']], params)
        tmppoints = gpd.GeoDataFrame(tmppoints)

    tmppoints.crs = shape.crs
    tmppoints = tmppoints[tmppoints.intersects(shape.unary_union)]
    grid = gpd.GeoDataFrame(tmppoints)
    return grid


def get_tra_grids(seg, traj_points, params_grid):
    traj_points_seg_i = traj_points[
        traj_points["timestamp"].between(
            seg["time_o"], seg["time_d"], inclusive="both"
        )
    ]
    traj_points_seg_i.sort_values('timestamp',inplace=True)
    if len(traj_points_seg_i) == 0:
        temp = pd.DataFrame(
            {
                "lat": [seg["lat_o"], seg["lat_d"]],
                "lng": [seg["lng_o"], seg["lng_d"]],
                "entity_id": traj_points["entity_id"].values[0],
            }
        )
        traj = tbd.points_to_traj(temp, col=["lng", "lat", "entity_id"])
        traj.set_crs(4326, inplace=True)
    elif len(traj_points_seg_i[['lat', 'lng']].drop_duplicates()) == 1:
        temp = pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "lat": seg["lat_o"],
                            "lng": seg["lng_o"],
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
                            "lat": seg["lat_d"],
                            "lng": seg["lng_d"],
                            "entity_id": traj_points_seg_i["entity_id"].values[0],
                        }
                    ]
                ),
            ]
        )
        traj = tbd.points_to_traj(temp, col=["lng", "lat", "entity_id"])
        traj.set_crs(4326, inplace=True)
    else:
        traj = tbd.points_to_traj(traj_points_seg_i, col=[
                                  "lng", "lat", "entity_id"])
        traj.set_crs(4326, inplace=True)

    tra_grids = traj_to_grids(traj, params=params_grid)
    if len(tra_grids) < 2:
        return traj, tra_grids, 0, 0, 0, 0, 0, 0, 0

    tra_grids["grid_lng"], tra_grids["grid_lat"] = tbd.grid_to_centre(
        [tra_grids["loncol_1"], tra_grids["loncol_2"], tra_grids["loncol_3"]], params_grid)
    tra_grids["grid"] = tra_grids["loncol_1"].astype(
        str) + "_" + tra_grids["loncol_2"].astype(str) + "_" + tra_grids["loncol_3"].astype(str)
    dis = list()
    traj_prjPt_lng = list()
    traj_prjPt_lat = list()
    for index, _grid in tra_grids.iterrows():
        # (traj["geometry"].values[0].distance(_grid["geometry"].centroid))
        traj_prjPt = ops.nearest_points(
            traj["geometry"].values[0], _grid['geometry'].centroid)[0]
        traj_prjPt_lng.append(traj_prjPt.x)
        traj_prjPt_lat.append(traj_prjPt.y)
        dis.append(traj["geometry"].values[0].line_locate_point(
            traj_prjPt, True))
    tra_grids["traj_prjPt_lng"] = traj_prjPt_lng
    tra_grids["traj_prjPt_lat"] = traj_prjPt_lat
    tra_grids["sort"] = dis
    tra_grids["entity_id"] = traj["entity_id"].values[0]
    tra_grids.sort_values(by="sort", inplace=True)

    _loncol_1_min = tra_grids['loncol_1'].min()
    _loncol_1_max = tra_grids['loncol_1'].max()
    _loncol_2_min = tra_grids['loncol_2'].min()
    _loncol_2_max = tra_grids['loncol_2'].max()
    _loncol_3_min = tra_grids['loncol_3'].min()
    _loncol_3_max = tra_grids['loncol_3'].max()

    gridslist = tra_grids[['loncol_1', 'loncol_2', 'loncol_3']].values.tolist()

    return traj, tra_grids, _loncol_1_min, _loncol_1_max, _loncol_2_min, _loncol_2_max, _loncol_3_min, _loncol_3_max, gridslist


def _get_var_len(tra_grids, dur_nsp):
    _len = basicfunc.getlinestringlen(LineString(
        tra_grids[["traj_prjPt_lng", "traj_prjPt_lat"]].values))
    _speed = _len/dur_nsp
    return np.round(_len, 2), np.round(_speed, 2)


def _get_var_dir(tra_grids):
    _tra_grids = tra_grids.copy()
    o_point = [_tra_grids.head(1).iloc[0]["traj_prjPt_lng"],
               _tra_grids.head(1).iloc[0]["traj_prjPt_lat"]]
    d_point = [_tra_grids.tail(1).iloc[0]["traj_prjPt_lng"],
               _tra_grids.tail(1).iloc[0]["traj_prjPt_lat"]]
    od_head = basicfunc.LatLng2Degree(
        o_point[1],
        o_point[0],
        d_point[1],
        d_point[0],
    )

    _tra_grids[["traj_prjPt_lng_next", "traj_prjPt_lat_next"]] = _tra_grids[[
        "traj_prjPt_lng", "traj_prjPt_lat"]].shift(-1)
    _tra_grids = _tra_grids[:-1]
    _tra_grids[["_head", "_turn"]] = _tra_grids[["traj_prjPt_lng", "traj_prjPt_lat",
                                                 "traj_prjPt_lng_next", "traj_prjPt_lat_next"]].apply(
        lambda tra_grid: _get_dir_ht(od_head, d_point, tra_grid), axis=1, result_type="expand")

    dir_head = np.nan_to_num(_tra_grids["_head"].mean())
    dir_turn = np.nan_to_num(
        (abs(_tra_grids["_turn"] - _tra_grids["_turn"].shift())).iloc[1:-1].mean())
    return np.round(dir_head, 2), np.round(dir_turn, 2)


def _get_dir_ht(_od_head, _d_point, _tra_grid):
    _h = basicfunc.LatLng2Degree(
        _tra_grid["traj_prjPt_lat"],
        _tra_grid["traj_prjPt_lng"],
        _d_point[1],
        _d_point[0],
    )
    _t = basicfunc.LatLng2Degree(
        _tra_grid["traj_prjPt_lat"],
        _tra_grid["traj_prjPt_lng"],
        _tra_grid["traj_prjPt_lat_next"],
        _tra_grid["traj_prjPt_lng_next"],
    )
    return abs(_od_head - _h), abs(_od_head - _t)


def _get_var_det(seg, traj, tra_grids, detectors_inarea):
    tra_grids_loncol_1_min = tra_grids["loncol_1"].min()
    tra_grids_loncol_1_max = tra_grids["loncol_1"].max()
    tra_grids_loncol_2_min = tra_grids["loncol_2"].min()
    tra_grids_loncol_2_max = tra_grids["loncol_2"].max()
    tra_grids_loncol_3_min = tra_grids["loncol_3"].min()
    tra_grids_loncol_3_max = tra_grids["loncol_3"].max()

    detectors_inarea_camera = detectors_inarea[
        detectors_inarea["type"].isin(["camera"])
    ].copy()

    detectors = detectors_inarea_camera.drop(
        index=list(set(detectors_inarea_camera.index.tolist()).intersection(set([seg["det_name_o"], seg["det_name_d"]]))))

    _det_count_ = list()
    _det_dist_ = list()
    _det_len_ = list()
    _detectors = detectors.query(
        f"((loncol_1 <={tra_grids_loncol_1_max}) and (loncol_1 >={tra_grids_loncol_1_min})) and (loncol_2 <={tra_grids_loncol_2_max}) and ((loncol_2 >={tra_grids_loncol_2_min})) and ((loncol_3 <={tra_grids_loncol_3_max}) and (loncol_3 >={tra_grids_loncol_3_min}))"
    )
    _detectors_ = _detectors.set_geometry("point", crs=4326).to_crs(3395)

    if len(_detectors_) == 0:
        det_count = det_dist = det_len = 0
    else:
        for detname, det in _detectors_.iterrows():
            inter = set(det["grids"]).intersection(
                set(tra_grids["grid"].tolist()))
            if len(inter) > 0:
                _det_count_.append(detname)
                _det_dist_.append(len(inter) / len(det["grids"]))
                _det_len_.append(
                    traj.to_crs(3395)["geometry"].values[0].distance(det["point"]))
    det_count = len(_det_count_)
    det_dist = np.nan_to_num(np.sum(_det_dist_))
    det_len = np.nan_to_num(np.sum(_det_len_))
    return np.round(det_count, 2), np.round(det_dist, 2), np.round(det_len, 2)


def _get_ob_var(seg, traj_points, detectors_inarea, params_grid):
    if seg.dur_nsp.seconds == 0 or (seg.det_name_o == seg.det_name_d):
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "ignore"

    traj, tra_grids, loncol_1_min, loncol_1_max, loncol_2_min, loncol_2_max, loncol_3_min, loncol_3_max, gridslist = get_tra_grids(
        seg, traj_points, params_grid)
    if len(tra_grids) < 2:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "ignore"

    _length, _speed = _get_var_len(tra_grids, seg.dur_nsp.seconds)
    _dir_head, _dir_turn = _get_var_dir(tra_grids)
    _det_count, _det_dist, _det_len = _get_var_det(
        seg, traj, tra_grids, detectors_inarea)
    return _length, _speed, _dir_head, _dir_turn, _det_count, _det_dist, _det_len, loncol_1_min, loncol_1_max, loncol_2_min, loncol_2_max, loncol_3_min, loncol_3_max, gridslist, traj["geometry"].values[0], "gotten"


def get_ST_seg_ob_var(ST_seg, traj_points, detectors_inarea, params_hexa):
    ST_seg_ob_var = pd.DataFrame()
    # for id_seg, tlseg in ST_seg.groupby("tlSeg"):
    #    ST_seg.loc[tlseg.index, "index"] = list(range(1, len(tlseg) + 1))
    ST_seg_ob_var[["tlSeg", "object", "pathset", "det_name_o", "det_name_d", "det_area_o", "det_area_d", "time_o", "time_d", "dur_nsp", "dist_nsp"]
                  ] = ST_seg[["tlSeg", "object", "pathset", "det_name_o", "det_name_d", "det_area_o", "det_area_d", "time_o", "time_d", "dur_nsp", "dist_nsp"]]
    ST_seg_ob_var[["path_length", "path_speed", "path_dir_head", "path_dir_turn", "path_det_count", "path_det_dist", "path_det_len", "loncol_1_min", "loncol_1_max", "loncol_2_min", "loncol_2_max", "loncol_3_min", "loncol_3_max", "gridslist",
                   "true_traj", "ob_var_note"]] = ST_seg.parallel_apply(lambda seg: _get_ob_var(seg, traj_points, detectors_inarea, params_hexa), axis=1, result_type="expand")

    # ST_seg_ob_var.append(tlseg[["tlSeg", "index", "object", "pathset", "det_name_o", "det_name_d", "dur_nsp", "dist_nsp",
    #                        "path_length", "path_speed", "path_dir_head", "path_dir_turn", "path_det_count", "path_det_dist", "path_det_len", "ob_var_note"]])
    return ST_seg_ob_var  # pd.concat(ST_seg_ob_var)
