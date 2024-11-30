import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import transbigdata as tbd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from . import basicfunc, creategraph, stats, util_coord, utils_graph


def _get_var_len(pathroads, lp_duration):
    # lp_len = pathroads["length"].sum()####
    lp_len = pathroads["_length_"].sum()  ####
    lp_speed = lp_len / lp_duration
    return lp_len, lp_speed


def _get_dir_ht(_pathroad, network_nodes_info, op, _od_head, from_name, to_name):
    _h = basicfunc.LatLng2Degree(
        network_nodes_info.loc[op]["y"],
        network_nodes_info.loc[op]["x"],
        network_nodes_info.loc[_pathroad[to_name]]["y"],
        network_nodes_info.loc[_pathroad[to_name]]["x"],
    )
    _t = basicfunc.LatLng2Degree(
        network_nodes_info.loc[_pathroad[from_name]]["y"],
        network_nodes_info.loc[_pathroad[from_name]]["x"],
        network_nodes_info.loc[_pathroad[to_name]]["y"],
        network_nodes_info.loc[_pathroad[to_name]]["x"],
    )

    return abs(_od_head - _h), _t


def _get_var_dir(pathroads, lp, network_nodes_info, from_name, to_name):
    op = lp[0]
    dp = lp[-1]
    _od_head = basicfunc.LatLng2Degree(
        network_nodes_info.loc[op]["y"],
        network_nodes_info.loc[op]["x"],
        network_nodes_info.loc[dp]["y"],
        network_nodes_info.loc[dp]["x"],
    )
    # od_head = abs(360 - _od_head) if _od_head > 180 else _od_head

    _head = list()
    _turn = list()
    pathroads[["_head", "_turn"]] = pathroads.apply(
        lambda _pathroad: _get_dir_ht(
            _pathroad, network_nodes_info, op, _od_head, from_name, to_name
        ),
        axis=1,
        result_type="expand",
    )
    lp_dir_head = pathroads["_head"].mean()
    lp_dir_turn = (
        (abs(pathroads["_turn"] - pathroads["_turn"].shift())).fillna(0)
    ).mean()
    return lp_dir_head, lp_dir_turn


def ___get_var_dir(pathroads, lp, roadsnetworks, from_name, to_name):
    nodes_gdf = utils_graph.graph_to_gdfs(roadsnetworks, edges=False)
    op = lp[0]
    dp = lp[-1]
    _od_head = basicfunc.LatLng2Degree(
        nodes_gdf.loc[op]["y"],
        nodes_gdf.loc[op]["x"],
        nodes_gdf.loc[dp]["y"],
        nodes_gdf.loc[dp]["x"],
    )
    # od_head = abs(360 - _od_head) if _od_head > 180 else _od_head

    _head = list()
    _turn = list()
    for i, _pathroad in pathroads.iterrows():
        _h = basicfunc.LatLng2Degree(
            nodes_gdf.loc[op]["y"],
            nodes_gdf.loc[op]["x"],
            nodes_gdf.loc[_pathroad[to_name]]["y"],
            nodes_gdf.loc[_pathroad[to_name]]["x"],
        )
        # _head.append(abs(360 - _h) if _h > 180 else _h)
        _head.append(abs(_od_head - _h))

        _t = basicfunc.LatLng2Degree(
            nodes_gdf.loc[_pathroad[from_name]]["y"],
            nodes_gdf.loc[_pathroad[from_name]]["x"],
            nodes_gdf.loc[_pathroad[to_name]]["y"],
            nodes_gdf.loc[_pathroad[to_name]]["x"],
        )
        # _turn.append(abs(360 - _t) if _t > 180 else _t)
        _turn.append(_t)

    pathroads["_head"] = _head
    pathroads["_turn"] = _turn
    lp_dir_head = pathroads["_head"].mean()
    lp_dir_turn = (
        (abs(pathroads["_turn"] - pathroads["_turn"].shift())).fillna(0)
    ).mean()
    return lp_dir_head, lp_dir_turn


def ____get_var_det(pathroads, detectors_od, detectors):
    inter = gpd.overlay(pathroads, detectors, "intersection")
    _detectors = gpd.GeoDataFrame(
        detectors["name"],
        geometry=gpd.points_from_xy(detectors.lng, detectors.lat),
        crs="epsg:4326",
    ).to_crs("epsg:3395")

    _det_count = 0
    _det_dist = list()
    _det_len = list()
    for _name_det, _inters_uns in inter.groupby("name"):
        if _name_det in detectors_od:
            continue
        _det_count += 1

        _inters_uns_ = _inters_uns.copy().to_crs("epsg:3395")
        _det_dist.append(
            _inters_uns_.apply(
                lambda x: np.min(
                    _detectors[_detectors["name"].isin([_name_det])]
                    .distance(x["geometry"])
                    .tolist()
                )
                if x["geometry"].type == "LineString"
                else np.min(
                    [
                        _detectors[_detectors["name"].isin([_name_det])]
                        .distance(line)
                        .tolist()
                        for line in x["geometry"].geoms
                    ]
                ),
                axis=1,
            ).mean()
        )
        _det_len.append(
            _inters_uns.apply(
                lambda x: basicfunc.getlinestringlen(x["geometry"])
                if x["geometry"].type == "LineString"
                else np.max(
                    [basicfunc.getlinestringlen(line) for line in x["geometry"].geoms]
                ),
                axis=1,
            ).mean()
        )

    lp_det_count = _det_count
    lp_det_dist = np.nan_to_num(np.mean(_det_dist))
    lp_det_len = np.nan_to_num(np.mean(_det_len))

    return lp_det_count, lp_det_dist, lp_det_len


def ___get_det_inter(pathroad, detectors, bound_grid):
    _det_count_ = list()
    _det_dist_ = list()
    _det_len_ = list()
    for p in pathroad["edge_grid"]:
        path_inter_detector = gpd.overlay(
            bound_grid.loc[p],
            detectors,
            "intersection",
        )
        _det_count_.append(len(path_inter_detector))

        # if len(path_inter_detector)>0:
    #     _det_dist_.append(
    #         tbd.getdistance(
    #             detectors.loc[path_inter_detector["name"]]["lng"].values,
    #             detectors.loc[path_inter_detector["name"]]["lat"].values,
    #             path_inter_detector["geometry"].centroid.x.values,
    #             path_inter_detector["geometry"].centroid.y.values,
    #         ).mean()
    #     )

    #     _det_len_.append(
    #         (
    #             path_inter_detector["geometry"].area.values
    #             / detectors.loc[path_inter_detector["name"]]["geometry"].area.values
    #         ).mean()
    #     )
    # else:
    #     _det_dist_.append(0)
    #     _det_len_.append(0)

    return (
        np.mean(np.nan_to_num(_det_count_)),
        0,
        0
        # np.mean(np.nan_to_num(_det_dist_)),
        # np.mean(np.nan_to_num(_det_len_)),
    )


def flatten(li):
    return sum(([x] if not isinstance(x, list) else flatten(x) for x in li), [])


def __get_det_inter(pathroad, detectors, bound_grid):
    path_inter_detector = gpd.overlay(
        bound_grid.loc[flatten(pathroad["edge_grid"])],
        detectors.reset_index(),
        "intersection",
    )
    _det_count_ = len(path_inter_detector) / len(pathroad["edge_grid"])

    _det_dist_ = tbd.getdistance(
        detectors.loc[path_inter_detector["name"]]["lng"].values,
        detectors.loc[path_inter_detector["name"]]["lat"].values,
        path_inter_detector["geometry"].centroid.x.values,
        path_inter_detector["geometry"].centroid.y.values,
    ).mean()

    _det_len_ = (
        path_inter_detector["geometry"].area.values
        / detectors.loc[path_inter_detector["name"]]["geometry"].area.values
    ).mean()

    return (
        np.nan_to_num(_det_count_),
        np.nan_to_num(_det_dist_),
        np.nan_to_num(_det_len_),
    )


def _get_det_inter(pathroad, detectors):
    _det_count_ = list()
    _det_dist_ = list()
    _det_len_ = list()
    pathroad_count = len(pathroad["edge_grid"])
    _detectors = detectors.query(
        f"((LONCOL_min <={pathroad['LONCOL_max']}) and (LONCOL_max >={pathroad['LONCOL_min']})) and ((LATCOL_min <={pathroad['LATCOL_max']}) and (LATCOL_max >={pathroad['LATCOL_min']}))"
    )

    if len(_detectors) == 0:
        return 0, 0, 0
    
    for detname, det in _detectors.iterrows():
        for road in pathroad["edge_grid"]:
            inter = set(det["grids"]).intersection(set(road))     
            if len(inter) > 0:
                _det_count_.append(detname)
                _det_dist_.append(len(inter) / len(det["grids"]))
                _det_len_.append(det["geometry"].distance(pathroad["geometry"]))
                
    return (
        len(_det_count_) / pathroad_count,
        np.nan_to_num(np.mean(_det_dist_) / pathroad_count),
        np.nan_to_num(np.mean(_det_len_) / pathroad_count),
    )


def _get_var_det(pathroads, detectors):
    temp = pathroads.apply(
        lambda pathroad: _get_det_inter(pathroad, detectors),
        axis=1,
        result_type="expand",
    )

    return temp[0].sum(), temp[1].sum(), temp[2].sum()


def getpathroadgridbound(pathroad):
    pathroadgridlist = flatten(pathroad["edge_grid"])
    pathroad_LONCOL_list = [
        int(pathroadgrid.split("_")[0]) for pathroadgrid in pathroadgridlist
    ]
    pathroad_LATCOL_list = [
        int(pathroadgrid.split("_")[1]) for pathroadgrid in pathroadgridlist
    ]
    return (
        np.min(pathroad_LONCOL_list),
        np.max(pathroad_LONCOL_list),
        np.min(pathroad_LATCOL_list),
        np.max(pathroad_LATCOL_list),
    )


def pe_basicvars(
    network_nodes_info,
    network_edges_info,
    lp,
    lp_len_eulid,
    lp_duration,
    detectors,
    bound_grid,
    from_name,
    to_name,
):
    #route_list = list()

    #for u, v in zip(lp[:-1], lp[1:]):
        # if there are parallel edges, select the shortest in length
    #    pathroad = min(
    #        roadsnetworks.get_edge_data(u, v).values(), key=lambda r: r["_length_"]
    #    )
    #    route_list.append(pathroad)
    #lp_paths = gpd.GeoDataFrame(
    #    pd.DataFrame(route_list), crs=roadsnetworks.graph["crs"]
    #)
    lp_paths = network_edges_info.loc[list(zip(lp[:-1], lp[1:]))]
    lp_paths[
        [
            "LONCOL_min",
            "LONCOL_max",
            "LATCOL_min",
            "LATCOL_max",
        ]
    ] = lp_paths.apply(
        lambda pathroad: getpathroadgridbound(pathroad),
        axis=1,
        result_type="expand",
    )
    lp_paths = lp_paths.to_crs(3395)
    
    try:
        lp_len, lp_speed = _get_var_len(lp_paths, lp_duration)
    except:
        print("error_get_var_len")
        return
    try:
        lp_dir_head, lp_dir_turn = _get_var_dir(
            lp_paths, lp, network_nodes_info, from_name, to_name
        )
    except:
        print("error_get_var_dir")
        return
    try:
        lp_det_count, lp_det_dist, lp_det_len = _get_var_det(lp_paths, detectors)

    except:
        print("error_get_var_det")
        return

    return (
        lp_paths,
        lp_len,
        lp_speed,
        lp_dir_head,
        lp_dir_turn,
        lp_det_count,
        lp_det_dist,
        lp_det_len,
    )