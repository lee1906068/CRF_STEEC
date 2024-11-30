from itertools import product

import geopandas as gpd
import geopy.distance
import networkx as nx
import numpy as np
import pandas as pd
import psycopg2
import transbigdata as tbd
from shapely import unary_union, wkt
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from . import (
    basicfunc,
    creategraph,
    distance,
    logs,
    settings,
    stats,
    util_coord,
    utils_graph,
)


def getbasicdata(
    _conn,
    _sampled_bounds,
    _config,
):
    _bounds_buffer = _config["_bounds_buffer"]
    _grid_accuracy = _config["_grid_accuracy"]
    _grid_cluster_accuracy = _config["_grid_cluster_accuracy"]

    bound_min = [
        geopy.distance.distance(kilometers=_bounds_buffer)
        .destination((_sampled_bounds[1], _sampled_bounds[0]), bearing=225)
        .longitude,
        geopy.distance.distance(kilometers=_bounds_buffer)
        .destination((_sampled_bounds[1], _sampled_bounds[0]), bearing=225)
        .latitude,
    ]
    bound_max = [
        geopy.distance.distance(kilometers=_bounds_buffer)
        .destination((_sampled_bounds[3], _sampled_bounds[2]), bearing=45)
        .longitude,
        geopy.distance.distance(kilometers=_bounds_buffer)
        .destination((_sampled_bounds[3], _sampled_bounds[2]), bearing=45)
        .latitude,
    ]
    bound = [bound_min[0], bound_min[1], bound_max[0], bound_max[1]]

    bound_polygon_str = f"POLYGON(({bound_min[0]} {bound_min[1]},{bound_min[0]} {bound_max[1]},{bound_max[0]} {bound_max[1]},{bound_max[0]} {bound_min[1]},{bound_min[0]} {bound_min[1]}))"
    st_geomfromtext_str = f"ST_GeomFromText('{bound_polygon_str}',4326)"
    st_contains_str = f"ST_Contains({st_geomfromtext_str},geometry)"

    sql1 = f"SELECT * FROM {_config['_roadnet_nodes']} where {st_contains_str}"
    sql2 = f"SELECT * FROM {_config['_roadnet_edges']} where {st_contains_str}"

    sql3 = f"SELECT * FROM {_config['_dt_roadnet_nodes']} where {st_contains_str}"
    sql4 = f"SELECT * FROM {_config['_dt_roadnet_edges']} where {st_contains_str}"
    sql5 = f"SELECT * FROM {_config['_dt_roads']} where {st_contains_str}"

    _roadnodes = gpd.GeoDataFrame.from_postgis(
        sql1, _conn, "geometry", crs="4326")
    _roadways = gpd.GeoDataFrame.from_postgis(
        sql2, _conn, "geometry", crs="4326")
    _det_nodes = gpd.GeoDataFrame.from_postgis(
        sql3, _conn, "geometry", crs="4326")
    _det_ways = gpd.GeoDataFrame.from_postgis(
        sql4, _conn, "geometry", crs="4326")
    detectors_inarea = gpd.GeoDataFrame.from_postgis(
        sql5, _conn, "geometry", crs="4326"
    )

    _roadnodes["ref"] = ""
    nodes_ = _roadnodes.append(_det_nodes).set_crs(4326)
    nodes_.set_index("osmid", inplace=True)

    ways_ = _roadways.append(_det_ways).set_crs(4326)
    ways_["from_to"] = ways_.parallel_apply(
        lambda way_: (way_["from"], way_["to"]), axis=1
    )
    ways_.set_index("from_to", inplace=True)
    #logs.info("get basicdata from database successfully")

    grid_bound, params_bound = tbd.area_to_grid(
        bound, accuracy=_grid_accuracy, method="rect", params="auto"
    )
    grid_bound["grid"] = (
        grid_bound["LONCOL"].astype(str) + "_" +
        grid_bound["LATCOL"].astype(str)
    )
    grid_bound = grid_bound.set_crs(settings.default_crs)
    grid_bound["type"] = _grid_accuracy

    grid_cluster_bound, params_cluster_bound = tbd.area_to_grid(
        bound, accuracy=_grid_cluster_accuracy, method="rect", params="auto"
    )
    grid_cluster_bound["grid"] = (
        grid_cluster_bound["LONCOL"].astype(str)
        + "_"
        + grid_cluster_bound["LATCOL"].astype(str)
    )
    grid_cluster_bound = grid_cluster_bound.set_crs(settings.default_crs)
    grid_cluster_bound["type"] = _grid_cluster_accuracy

    def detector_to_grid(detector, _grid_accuracy, params_bound):
        area_to_grid = tbd.area_to_grid(
            detector,
            accuracy=50,
            method="rect",
            params=params_bound,
        )[0]
        return (area_to_grid["LONCOL"].map(str) + "_" + area_to_grid["LATCOL"].map(str)).values.tolist(), area_to_grid[["LONCOL", "LATCOL"]].values.tolist(), area_to_grid["LONCOL"].min(), area_to_grid["LONCOL"].max(), area_to_grid["LATCOL"].min(), area_to_grid["LATCOL"].max()

    detectors_inarea[
        [
            "grids",
            "grids_list",
            "LONCOL_min",
            "LONCOL_max",
            "LATCOL_min",
            "LATCOL_max",
        ]
    ] = detectors_inarea.parallel_apply(
        lambda detector: detector_to_grid(gpd.GeoDataFrame([detector]), 50, params_bound), axis=1, result_type="expand",
    )

    #logs.info("bound grids successfully")

    return (
        bound,
        nodes_,
        ways_,
        _roadnodes,
        _roadways,
        detectors_inarea,
        grid_bound,
        params_bound,
        grid_cluster_bound,
        params_cluster_bound,
    )


def network_edge_geometry_grid_line(way_geometry, params_grid):
    roadline = pd.DataFrame(list(way_geometry.coords), columns=["lng", "lat"])
    roadline["LONCOL"], roadline["LATCOL"] = tbd.GPS_to_grid(
        roadline["lng"], roadline["lat"], params_grid
    )
    roadline["grid_lng"], roadline["grid_lat"] = tbd.grid_to_centre(
        [roadline["LONCOL"], roadline["LATCOL"]], params_grid
    )
    roadline["grid"] = (
        roadline["LONCOL"].astype(str) + "_" + roadline["LATCOL"].astype(str)
    )

    geometry = LineString(roadline[["grid_lng", "grid_lat"]].values.tolist())
    length_grid = round(basicfunc.getlinestringlen(geometry), 2)
    hash_geometry = tbd.geohash_encode(
        roadline["grid_lng"], roadline["grid_lat"], precision=12
    ).values.tolist()
    return geometry, length_grid, hash_geometry


def network_edge_grid(way_geometry_wkt, params_bound, accuracy):
    way_geometry = wkt.loads(way_geometry_wkt)
    if way_geometry.length > 0:
        edge_grid = tbd.area_to_grid(
            gpd.GeoDataFrame([way_geometry], columns=["geometry"]),
            accuracy=accuracy,
            method="rect",
            params=params_bound,
        )[0]
        dis = list()
        for index, _grid in edge_grid.iterrows():
            dis.append(way_geometry.project(_grid["geometry"].centroid, True))
        edge_grid["sort"] = dis
        edge_grid.sort_values(by="sort", inplace=True)
        return (
            edge_grid["LONCOL"].astype(
                str) + "_" + edge_grid["LATCOL"].astype(str)
        ).values.tolist()
    else:
        LONCOL_LATCOL = tbd.GPS_to_grid(
            way_geometry.centroid.x, way_geometry.centroid.y, params=params_bound
        )
        return [str(LONCOL_LATCOL[0]) + "_" + str(LONCOL_LATCOL[1])]


def waysinfo_update(way, nodes_info, params_bound, grid_accuracy):
    from_grid = dict(nodes_info[nodes_info["osmid"]
                     == way["from"]].iloc[0])["grid"]
    to_grid = dict(nodes_info[nodes_info["osmid"]
                   == way["to"]].iloc[0])["grid"]
    ignore = True if from_grid == to_grid else False
    geometry, length_grid, hash_geometry = network_edge_geometry_grid_line(
        way["geometry"], params_bound
    )

    edge_grid = network_edge_grid(
        way["geometry"].wkt, params_bound, accuracy=grid_accuracy
    )

    return (
        from_grid,
        to_grid,
        from_grid + "-" + to_grid,
        (from_grid, to_grid),
        edge_grid,
        geometry,
        hash_geometry,
        way["length"],
        length_grid,
        ignore,
    )


def get_network_grid_info(_nodes_, _ways_, params_bound, _config):
    _nodes = _nodes_.copy()
    _ways = _ways_.copy()
    _nodes.reset_index(inplace=True)
    _ways.reset_index(inplace=True)
    _ways.drop(columns="from_to", inplace=True)
    grid_accuracy = _config["_grid_accuracy"]

    nodes_info = gpd.GeoDataFrame()
    nodes_info[_nodes.drop(columns="geometry").columns] = _nodes[
        _nodes.drop(columns="geometry").columns
    ].copy()

    nodes_info["LONCOL"], nodes_info["LATCOL"] = tbd.GPS_to_grid(
        nodes_info["lng"], nodes_info["lat"], params_bound
    )
    nodes_info["geometry"] = tbd.grid_to_polygon(
        [nodes_info["LONCOL"], nodes_info["LATCOL"]], params_bound
    )
    nodes_info["lng_grid"], nodes_info["lat_grid"] = tbd.grid_to_centre(
        [nodes_info["LONCOL"], nodes_info["LATCOL"]], params_bound
    )
    nodes_info["grid"] = (
        nodes_info["LONCOL"].astype(str) + "_" +
        nodes_info["LATCOL"].astype(str)
    )

    network_grid_nodes_info = gpd.GeoDataFrame()
    for col in nodes_info.columns:
        if col in _nodes.drop(columns="geometry").columns:
            network_grid_nodes_info[col] = (
                nodes_info.groupby(["grid"])[col].apply(
                    lambda x: list(x)).to_frame()
            )
        else:
            network_grid_nodes_info[col] = (
                nodes_info.groupby(["grid"])[col].apply(
                    lambda x: x.iloc[0]).to_frame()
            )
    network_grid_nodes_info["x"] = network_grid_nodes_info["lng_grid"]
    network_grid_nodes_info["y"] = network_grid_nodes_info["lat_grid"]

    network_grid_nodes_info.drop(columns="grid", inplace=True)
    network_grid_nodes_info = network_grid_nodes_info.set_crs(
        settings.default_crs)

    #logs.info("get network_grid_nodes_info successfully")

    ways_info = gpd.GeoDataFrame()
    ways_info[_ways.drop(columns=["geometry", "length"]).columns] = _ways[
        _ways.drop(columns=["geometry", "length"]).columns
    ].copy()

    ways_info[
        [
            "from_grid",
            "to_grid",
            "edgeid",
            "from_to",
            "edge_grid",
            "geometry",
            "hash_geometry",
            "_length",
            "length",
            "ignore",
        ]
    ] = _ways.apply(  # _ways.parallel_apply(
        lambda way: waysinfo_update(
            way, nodes_info, params_bound, grid_accuracy),
        axis=1,
        result_type="expand",
    )

    network_grid_edges_info = gpd.GeoDataFrame()
    for col in ways_info.columns:
        if col == "_length":
            network_grid_edges_info[col] = (
                ways_info.groupby(["from_to"])[col].apply(
                    lambda x: list(x)).to_frame()
            )
        elif col in _ways.drop(columns=["geometry", "length"]).columns:
            network_grid_edges_info[col] = (
                ways_info.groupby(["from_to"])[col].apply(
                    lambda x: list(x)).to_frame()
            )
        else:
            network_grid_edges_info[col] = (
                ways_info.groupby(["from_to"])[col]
                .apply(lambda x: x.iloc[0])
                .to_frame()
            )

    network_grid_edges_info.drop(columns="from_to", inplace=True)
    network_grid_edges_info = network_grid_edges_info.set_crs(
        settings.default_crs)

    #logs.info("get network_grid_edges_info successfully")

    return nodes_info, network_grid_nodes_info, ways_info, network_grid_edges_info


def _mergenearnodes(cl_nodes_info, cl_edges_info, buffer_mergenearnodes=60):
    grid_cl = set(cl_nodes_info["grid_cl"].values.tolist())
    grid_cl_list = list()
    flagflag = False
    for _cl in grid_cl:
        grid_cl_dict = dict()
        group_list = list()
        group_status = list()
        _cl_nodes_all = set(
            cl_nodes_info[cl_nodes_info["grid_cl"] == _cl].index)
        _cl_subset = cl_edges_info.query(
            f"(from_grid_cl=='{_cl}' and to_grid_cl=='{_cl}')"
        )
        for way_id, way in _cl_subset.iterrows():
            if way["length"] > buffer_mergenearnodes or way["ignore"]:
                continue
            else:
                group_list.append({way["from_grid"], way["to_grid"]})
                group_status.append(1)

        number_of_valid_groups = sum(group_status)
        while True:
            for group_no1, group1 in enumerate(group_list):
                if group_status[group_no1] == 0:
                    continue
                group1_count = len(group1)
                while True:
                    for group_no2, group2 in enumerate(group_list):
                        if group_status[group_no2] == 0:
                            continue
                        if group_no1 == group_no2:
                            continue
                        if len(group1.intersection(group2)) > 0:
                            group1.update(group2)
                            group_status[group_no2] = 0

                    new_group1_count = len(group1)
                    if group1_count == new_group1_count:
                        break
                    else:
                        group1_count = len(group1)

            new_number_of_valid_groups = sum(group_status)
            if number_of_valid_groups == new_number_of_valid_groups:
                break
            else:
                number_of_valid_groups = new_number_of_valid_groups

        max_intersection_id = 1
        flag = False
        for group_no, group in enumerate(group_list):
            if group_status[group_no] == 0:
                continue
            for node in group:
                grid_cl_dict[node] = _cl + "_" + str(max_intersection_id)
                flag = True
            if flag:
                max_intersection_id += 1
                flag = False

        for _idx, _is in enumerate(_cl_nodes_all - grid_cl_dict.keys()):
            grid_cl_dict[_is] = _cl + "_" + str(_idx + 1) + "_is"
        grid_cl_list.append(grid_cl_dict)

    #logs.info(f"mergenearnodes road cluster complete")
    cl_nodes_info["mergenearnodes"] = pd.concat(
        [
            pd.DataFrame.from_dict(_grid_cl_list, orient="index")
            for _grid_cl_list in grid_cl_list
        ]
    )
    return cl_nodes_info


def _nodesinfo_update(node, params_bound):
    if len(node["grid"]) > 1:
        LONCOL, LATCOL = tbd.GPS_to_grid(
            np.mean(node["lng_grid"]), np.mean(node["lat_grid"]), params_bound
        )
        return LONCOL, LATCOL, str(LONCOL) + "_" + str(LATCOL)
    else:
        return node["LONCOL"][0], node["LATCOL"][0], node["grid"][0]


def _waysinfo_update(
    edge,
    network_grid_cl_nodes_info,
    cl_grid_new,
    cl_grid_org,
    params_bound,
    grid_accuracy,
):
    def qurey_cl_grid_new_cl_grid_org(qurey, grid_new, grid_org):
        for _ in zip(grid_new, grid_org):
            if qurey in _[1]:
                return _[0]

    from_grid_new = qurey_cl_grid_new_cl_grid_org(
        edge["from_grid"], cl_grid_new, cl_grid_org
    )
    to_grid_new = qurey_cl_grid_new_cl_grid_org(
        edge["to_grid"], cl_grid_new, cl_grid_org
    )
    ignore_cl = True if from_grid_new == to_grid_new else False

    from_grid_new_point = (
        network_grid_cl_nodes_info.loc[[
            from_grid_new]]["lng_grid_new"].iloc[0],
        network_grid_cl_nodes_info.loc[[
            from_grid_new]]["lat_grid_new"].iloc[0],
    )
    to_grid_new_point = (
        network_grid_cl_nodes_info.loc[[to_grid_new]]["lng_grid_new"].iloc[0],
        network_grid_cl_nodes_info.loc[[to_grid_new]]["lat_grid_new"].iloc[0],
    )
    from_grid_new_point_end_length = tbd.getdistance(
        from_grid_new_point[0],
        from_grid_new_point[1],
        edge["geometry"].coords[-1][0],
        edge["geometry"].coords[-1][1],
    )
    point_start_to_grid_new_length = tbd.getdistance(
        edge["geometry"].coords[0][0],
        edge["geometry"].coords[0][1],
        to_grid_new_point[0],
        to_grid_new_point[1],
    )
    if edge["length"] < round(from_grid_new_point_end_length, 2):
        geometry_cl = LineString(
            [from_grid_new_point] + list(edge["geometry"].coords))
    else:
        geometry_cl = LineString(
            [from_grid_new_point] + list(edge["geometry"].coords)[1:]
        )

    if edge["length"] < round(point_start_to_grid_new_length, 2):
        geometry_cl = LineString(
            list(geometry_cl.coords) + [to_grid_new_point])
    else:
        geometry_cl = LineString(list(geometry_cl.coords)[
                                 :-1] + [to_grid_new_point])

    edge_grid_new = network_edge_grid(
        geometry_cl.wkt, params_bound, accuracy=grid_accuracy
    )

    length_grid = round(basicfunc.getlinestringlen(geometry_cl), 2)

    return (
        from_grid_new,
        to_grid_new,
        from_grid_new + "-" + to_grid_new,
        (from_grid_new, to_grid_new),
        ignore_cl,
        geometry_cl,
        edge_grid_new,
        length_grid,
    )


def get_network_cl_grid_info(
    network_grid_nodes_info,
    network_grid_edges_info,
    params_cluster_bound,
    params_bound,
    _config,
    mergenearnodes=True,
):
    grid_accuracy = _config["_grid_accuracy"]
    buffer_mergenearnodes = _config["_buffer_mergenearnodes"]
    cl_nodes_info = gpd.GeoDataFrame()
    cl_nodes_info["grid"] = network_grid_nodes_info.index
    cl_nodes_info.set_index("grid", inplace=True)
    cl_nodes_info[["lng_grid", "lat_grid"]] = network_grid_nodes_info[
        ["lng_grid", "lat_grid"]
    ]
    cl_nodes_info[["LONCOL", "LATCOL"]
                  ] = network_grid_nodes_info[["LONCOL", "LATCOL"]]
    cl_nodes_info["LONCOL_cl"], cl_nodes_info["LATCOL_cl"] = tbd.GPS_to_grid(
        network_grid_nodes_info["lng_grid"],
        network_grid_nodes_info["lat_grid"],
        params_cluster_bound,
    )
    cl_nodes_info["lng_grid_cl"], cl_nodes_info["lat_grid_cl"] = tbd.grid_to_centre(
        [cl_nodes_info["LONCOL_cl"], cl_nodes_info["LATCOL_cl"]], params_cluster_bound
    )
    cl_nodes_info["grid_cl"] = (
        cl_nodes_info["LONCOL_cl"].astype(str)
        + "_"
        + cl_nodes_info["LATCOL_cl"].astype(str)
    )

    cl_edges_info = gpd.GeoDataFrame()
    cl_edges_info["from_to"] = network_grid_edges_info.index
    cl_edges_info.set_index("from_to", inplace=True)
    cl_edges_info[
        [
            "from_grid",
            "to_grid",
            "edgeid",
            "edge_grid",
            "geometry",
            "length",
            "_length",
            "ignore",
        ]
    ] = network_grid_edges_info[
        [
            "from_grid",
            "to_grid",
            "edgeid",
            "edge_grid",
            "geometry",
            "length",
            "_length",
            "ignore",
        ]
    ]
    cl_edges_info["from_grid_cl"] = cl_nodes_info.loc[cl_edges_info["from_grid"]][
        "grid_cl"
    ].values
    cl_edges_info["to_grid_cl"] = cl_nodes_info.loc[cl_edges_info["to_grid"]][
        "grid_cl"
    ].values

    if mergenearnodes:
        cl_nodes_info = _mergenearnodes(
            cl_nodes_info, cl_edges_info, buffer_mergenearnodes
        )
    else:
        cl_nodes_info["mergenearnodes"] = cl_nodes_info["grid_cl"]

    network_grid_cl_nodes_info = gpd.GeoDataFrame()
    for col in cl_nodes_info.reset_index().columns:
        if (
            col
            in cl_nodes_info.reset_index()
            .drop(
                columns=[
                    "LONCOL_cl",
                    "LATCOL_cl",
                    "lng_grid_cl",
                    "lat_grid_cl",
                    "grid_cl",
                ]
            )
            .columns
        ):
            network_grid_cl_nodes_info[col] = (
                cl_nodes_info.reset_index()
                .groupby(["mergenearnodes"])[col]
                .apply(lambda x: list(x))
                .to_frame()
            )
        else:
            network_grid_cl_nodes_info[col] = (
                cl_nodes_info.reset_index()
                .groupby(["mergenearnodes"])[col]
                .apply(lambda x: x.iloc[0])
                .to_frame()
            )
    network_grid_cl_nodes_info.drop(columns="mergenearnodes", inplace=True)

    network_grid_cl_nodes_info[
        ["LONCOL_new", "LATCOL_new", "grid_new"]
        # network_grid_cl_nodes_info.parallel_apply(
    ] = network_grid_cl_nodes_info.apply(
        lambda node: _nodesinfo_update(node, params_bound),
        axis=1,
        result_type="expand",
    )
    (
        network_grid_cl_nodes_info["lng_grid_new"],
        network_grid_cl_nodes_info["lat_grid_new"],
    ) = tbd.grid_to_centre(
        [
            network_grid_cl_nodes_info["LONCOL_new"],
            network_grid_cl_nodes_info["LATCOL_new"],
        ],
        params_bound,
    )

    network_grid_cl_nodes_info["x"] = network_grid_cl_nodes_info["lng_grid_new"]
    network_grid_cl_nodes_info["y"] = network_grid_cl_nodes_info["lat_grid_new"]

    network_grid_cl_nodes_info.set_index("grid_new", inplace=True)

    #logs.info("get network_grid_cl_nodes_info successfully")

    cl_grid_new = network_grid_cl_nodes_info.index.tolist()
    cl_grid_org = network_grid_cl_nodes_info["grid"].tolist()
    cl_edges_info[
        [
            "from_grid_new",
            "to_grid_new",
            "edgeid_new",
            "from_to_new",
            "ignore_cl",
            "geometry",
            "edge_grid_new",
            "length_grid",
        ]
    ] = cl_edges_info.apply(  # cl_edges_info.parallel_apply(
        lambda edge: _waysinfo_update(
            edge,
            network_grid_cl_nodes_info,
            cl_grid_new,
            cl_grid_org,
            params_bound,
            grid_accuracy,
        ),
        axis=1,
        result_type="expand",
    )

    network_grid_cl_edges_info = gpd.GeoDataFrame()
    for col in cl_edges_info.columns:
        if col not in [
            "from_grid_new",
            "to_grid_new",
            "edgeid_new",
            "edge_grid_new",
            "from_to",
            "geometry",
            "length",
            "ignore_cl",
        ]:
            network_grid_cl_edges_info[col] = (
                cl_edges_info.groupby(["from_to_new"])[col]
                .apply(lambda x: list(x))
                .to_frame()
            )
            # network_grid_edges_info[col]=network_grid_edges_info[col].apply(lambda x:str(x).replace('[','').replace(']',''))
        else:
            network_grid_cl_edges_info[col] = (
                cl_edges_info.groupby(["from_to_new"])[col]
                .apply(lambda x: x.iloc[0])
                .to_frame()
            )

    def flatten(li):
        return sum(([x] if not isinstance(x, list) else flatten(x) for x in li), [])

    network_grid_cl_edges_info["_length_"] = network_grid_cl_edges_info.parallel_apply(
        lambda edge: np.mean(flatten(edge["_length"])), axis=1
    )

    network_grid_cl_edges_info.drop(columns="from_to_new", inplace=True)
    network_grid_cl_edges_info = network_grid_cl_edges_info.set_crs(
        settings.default_crs
    )

    #logs.info("get network_grid_cl_edges_info successfully")

    return (
        cl_nodes_info,
        network_grid_cl_nodes_info,
        cl_edges_info,
        network_grid_cl_edges_info,
    )
