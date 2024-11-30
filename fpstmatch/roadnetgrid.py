from itertools import product
from functools import partial
import geopandas as gpd
import geopy.distance
import networkx as nx
import numpy as np
import pandas as pd
import psycopg2
import transbigdata as tbd
from shapely import unary_union, wkt, ops
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
import multiprocessing as mp

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


def getdetetorgrids(_detector, params_grid):
    bounds = _detector.bounds
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
    if params_grid['method'] == 'hexa':
        tmppoints['loncol_1'],\
            tmppoints['loncol_2'],\
            tmppoints['loncol_3'] = tbd.GPS_to_grid(
                tmppoints['lon'], tmppoints['lat'], params_grid)
        tmppoints = tmppoints[['loncol_1',
                               'loncol_2', 'loncol_3']].drop_duplicates()
        tmppoints["grid"] = tmppoints["loncol_1"].astype(
            str) + "_" + tmppoints["loncol_2"].astype(str) + "_" + tmppoints["loncol_3"].astype(str)
        return tmppoints["grid"].tolist(), \
            tmppoints[["loncol_1", "loncol_2", "loncol_3"]].values.tolist(),\
            tmppoints["loncol_1"].min(), tmppoints["loncol_1"].max(), \
            tmppoints["loncol_2"].min(), tmppoints["loncol_2"].max(),\
            tmppoints["loncol_3"].min(), tmppoints["loncol_3"].max()
    else:
        tmppoints['LONCOL'],\
            tmppoints['LATCOL'] = tbd.GPS_to_grid(
                tmppoints['lon'], tmppoints['lat'], params_grid)
        tmppoints = tmppoints[['LONCOL', 'loncol_3']].drop_duplicates()
        tmppoints["grid"] = tmppoints["LONCOL"].astype(
            str) + "_" + tmppoints["loncol_3"].astype(str)
        return tmppoints["grid"].tolist(), \
            tmppoints[["LONCOL", "LATCOL"]].values.tolist(),\
            tmppoints["LONCOL"].min(), tmppoints["LONCOL"].max(),\
            tmppoints["LATCOL"].min(), tmppoints["LATCOL"].max()


def getbasicdata(
    _conn,
    _sampled_bounds,
    _config,
):
    _bounds_buffer = _config["_bounds_buffer"]
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

    if _config["_params_grid"]['method'] == 'hexa':
        detectors_inarea[
            [
                "grids",
                "grids_list",
                "loncol_1_min",
                "loncol_1_max",
                "loncol_2_min",
                "loncol_2_max",
                "loncol_3_min",
                "loncol_3_max",
            ]
        ] = detectors_inarea.parallel_apply(
            lambda detector: getdetetorgrids(detector["geometry"], _config["_params_grid"]), axis=1, result_type="expand",
        )
    else:
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
            lambda detector: getdetetorgrids(detector, _config["_params_grid"]), axis=1, result_type="expand",
        )

    return (
        bound,
        nodes_,
        ways_,
        _roadnodes,
        _roadways,
        detectors_inarea
    )


def network_edge_geometry_grid_line(way_geometry, params_grid):
    roadline = pd.DataFrame(list(way_geometry.coords), columns=["lng", "lat"])
    if params_grid["method"] == 'hexa':
        roadline["loncol_1"], roadline["loncol_2"], roadline["loncol_3"] = tbd.GPS_to_grid(
            roadline["lng"], roadline["lat"], params_grid
        )
        roadline["grid_lng"], roadline["grid_lat"] = tbd.grid_to_centre(
            [roadline["loncol_1"], roadline["loncol_2"],
                roadline["loncol_3"]], params_grid
        )
        roadline["grid"] = (
            roadline["loncol_1"].astype(
                str) + "_" + roadline["loncol_2"].astype(str) + "_" + roadline["loncol_3"].astype(str)
        )
    else:
        roadline["LONCOL"], roadline["LATCOL"] = tbd.GPS_to_grid(
            roadline["lng"], roadline["lat"], params_grid
        )
        roadline["grid_lng"], roadline["grid_lat"] = tbd.grid_to_centre(
            [roadline["LONCOL"], roadline["LATCOL"]], params_grid
        )
        roadline["grid"] = (
            roadline["LONCOL"].astype(str) + "_" +
            roadline["LATCOL"].astype(str)
        )

    geometry = LineString(roadline[["grid_lng", "grid_lat"]].values.tolist())
    length_grid = round(basicfunc.getlinestringlen(geometry), 2)
    hash_geometry = tbd.geohash_encode(
        roadline["grid_lng"], roadline["grid_lat"], precision=12
    ).values.tolist()
    return geometry, length_grid, hash_geometry


def traj_to_grids(traj, params):
    if (type(traj) == list) | (type(traj) == tuple):
        shape = ''
        bounds = traj
    elif type(traj) == gpd.geodataframe.GeoDataFrame:
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
        (2 * np.pi * 6371004 * np.cos((lat1 + lat2) * np.pi / 360))
    deltaLat = params["gridsize"] * 360 / (2 * np.pi * 6371004)

    method = params['method']

    if (method == 'tri') | (method == 'hexa'):
        longapsize = np.arange(
            bounds[0], bounds[2]+deltaLon/3, deltaLon/3).size
        latgapsize = np.arange(
            bounds[1], bounds[3]+deltaLat/3, deltaLat/3).size
        if longapsize == 0 and latgapsize != 0:
            tmppoints = pd.DataFrame(
                np.array(
                    np.meshgrid(
                        np.array([bounds[0]]*latgapsize),
                        np.arange(bounds[1],
                                  bounds[3]+deltaLat/3,
                                  deltaLat/3))
                ).reshape(2, -1).T, columns=['lon', 'lat'])
        elif latgapsize == 0 and longapsize != 0:
            tmppoints = pd.DataFrame(
                np.array(
                    np.meshgrid(
                        np.arange(bounds[0],
                                  bounds[2]+deltaLon/3,
                                  deltaLon/3),
                        np.array([bounds[1]]*longapsize))
                ).reshape(2, -1).T, columns=['lon', 'lat'])
        else:
            tmppoints = pd.DataFrame(
                np.array(
                    np.meshgrid(
                        np.arange(bounds[0],
                                  bounds[2]+deltaLon/3,
                                  deltaLon/3),
                        np.arange(bounds[1],
                                  bounds[3]+deltaLat/3,
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

    if type(shape) != gpd.geodataframe.GeoDataFrame:
        grid = gpd.GeoDataFrame(tmppoints)
    else:
        tmppoints.crs = shape.crs
        tmppoints = tmppoints[tmppoints.intersects(shape.unary_union)]
        grid = gpd.GeoDataFrame(tmppoints)
        grid.reset_index(inplace=True)
    return grid


def network_edge_grid(way_geometry, params_grid):
    if way_geometry.length > 0:
        edge_grid = traj_to_grids(gpd.GeoDataFrame([way_geometry],
                                                   columns=["geometry"]), params_grid)
        dis = list()
        way_prjPt_lng = list()
        way_prjPt_lat = list()
        for index, _grid in edge_grid.iterrows():
            way_prjPt = ops.nearest_points(
                way_geometry, _grid['geometry'].centroid)[0]
            way_prjPt_lng.append(way_prjPt.x)
            way_prjPt_lat.append(way_prjPt.y)
            dis.append(way_geometry.line_locate_point(way_prjPt, True))
        edge_grid["sort"] = dis
        edge_grid["way_prjPt_lng"] = way_prjPt_lng
        edge_grid["way_prjPt_lat"] = way_prjPt_lat
        edge_grid.sort_values(by="sort", inplace=True)

        od_coords = list(way_geometry.coords)
        way_prjPt_point_list = edge_grid[[
            'way_prjPt_lng', 'way_prjPt_lat']].values.tolist()
        if way_prjPt_point_list[0] != list(od_coords[0]):
            way_prjPt_point_list = [od_coords[0]]+way_prjPt_point_list
        if way_prjPt_point_list[-1] != list(od_coords[-1]):
            way_prjPt_point_list = way_prjPt_point_list+[od_coords[-1]]

        way_prjPt_geom = LineString(way_prjPt_point_list)

        if params_grid["method"] == 'hexa':
            edge_grid_list = (edge_grid["loncol_1"].astype(str) + "_" + edge_grid["loncol_2"].astype(
                str) + "_" + edge_grid["loncol_3"].astype(str)).values.tolist()
        else:
            edge_grid_list = (edge_grid["LONCOL"].astype(str) + "_" +
                              edge_grid["LATCOL"].astype(str)).values.tolist()
        return edge_grid_list, way_prjPt_geom
    else:
        way_prjPt_geom = way_geometry
        if params_grid["method"] == 'hexa':
            loncol_1_loncol_2_loncol_3 = tbd.GPS_to_grid(
                way_geometry.centroid.x, way_geometry.centroid.y, params=params_grid
            )

            edge_grid_list = [str(loncol_1_loncol_2_loncol_3[0][0]) + "_" + str(
                loncol_1_loncol_2_loncol_3[1][0]) + "_" + str(loncol_1_loncol_2_loncol_3[2][0])]

        else:
            LONCOL_LATCOL = tbd.GPS_to_grid(
                way_geometry.centroid.x, way_geometry.centroid.y, params=params_grid
            )
            edge_grid_list = [str(LONCOL_LATCOL[0]) +
                              "_" + str(LONCOL_LATCOL[1])]
        return edge_grid_list, way_prjPt_geom


def waysinfo_update(way, nodes_info, params_bound):
    from_grid = nodes_info[nodes_info["osmid"]
                           == way["from"]]["grid"].values[0]
    to_grid = nodes_info[nodes_info["osmid"]
                         == way["to"]]["grid"].values[0]
    ignore = True if from_grid == to_grid else False
    # geometry, length_grid, hash_geometry = network_edge_geometry_grid_line(
    #    way["geometry"], params_bound
    # )

    edge_grid, way_prjPt_geom = network_edge_grid(
        way["geometry"], params_bound
    )

    return (
        from_grid,
        to_grid,
        from_grid + "-" + to_grid,
        (from_grid, to_grid),
        edge_grid,
        way_prjPt_geom,
        # length_grid,
        # hash_geometry,
        ignore,
    )


def _merge_nodes_grid_(groupby_grid, params):
    group_name, group_data = groupby_grid
    rel = list()
    for col in group_data.columns:
        if col not in ['geometry', 'lng_grid', 'lat_grid', 'loncol_1', 'loncol_2', 'loncol_3']:
            rel.append(list(group_data[col]))
    rel.append(tbd.grid_to_polygon([group_data.iloc[0]["loncol_1"],
               group_data.iloc[0]["loncol_2"], group_data.iloc[0]["loncol_3"]], params)[0])
    rel.append(list(group_data['lng_grid'])[0])
    rel.append(list(group_data['lat_grid'])[0])
    rel.append(list(group_data['loncol_1'])[0])
    rel.append(list(group_data['loncol_2'])[0])
    rel.append(list(group_data['loncol_3'])[0])
    rel.append(group_name)

    return tuple(_rel for _rel in rel)


def _merge_edges_grid_(groupby_grid):
    group_name, group_data = groupby_grid
    rel = list()
    for col in group_data.columns:
        if col not in ['geometry', "way_prjPt_geom", 'length', "from_grid", "to_grid", "from_to", "edgeid", "ignore"]:
            rel.append(list(group_data[col]))
    rel.append(MultiLineString(list(group_data['geometry'])))
    rel.append(MultiLineString(list(group_data['way_prjPt_geom'])))
    rel.append(group_data['length'].mean())
    rel.append(list(group_data['from_grid'])[0])
    rel.append(list(group_data['to_grid'])[0])
    rel.append(list(group_data['from_to'])[0])
    rel.append(list(group_data['edgeid'])[0])
    rel.append(list(group_data['ignore'])[0])
    rel.append(group_name)

    return tuple(_rel for _rel in rel)


def get_network_grid_info(_nodes_, _ways_, _config):
    _nodes = _nodes_.copy()
    _ways = _ways_.copy()
    _nodes.reset_index(inplace=True)
    _ways.reset_index(inplace=True)
    _ways.drop(columns="from_to", inplace=True)

    nodes_info = gpd.GeoDataFrame()
    nodes_info = _nodes.copy()
    # nodes_info[_nodes.drop(columns="geometry").columns] = _nodes[
    #    _nodes.drop(columns="geometry").columns
    # ].copy()
    if _config['_params_grid']['method'] == 'hexa':
        def getgrid(_lng, _lat, _params):
            loncol_1_loncol_2_loncol_3 = tbd.GPS_to_grid(
                _lng, _lat, _params
            )
            loncol_1, loncol_2, loncol_3 = (loncol[0]
                                            for loncol in loncol_1_loncol_2_loncol_3)
            # geometry = tbd.grid_to_polygon(
            #    loncol_1_loncol_2_loncol_3, _params)[0]
            lng_grid, lat_grid = tbd.grid_to_centre(
                loncol_1_loncol_2_loncol_3, _params)
            grid = '_'.join([str(x[0]) for x in loncol_1_loncol_2_loncol_3])
            # return loncol_1, loncol_2, loncol_3, geometry, lng_grid[0], lat_grid[0], grid
            return loncol_1, loncol_2, loncol_3, lng_grid[0], lat_grid[0], grid

        nodes_info[['loncol_1', 'loncol_2', 'loncol_3', 'lng_grid', 'lat_grid', 'grid']] = nodes_info.parallel_apply(
            lambda x: getgrid(x['lng'], x['lat'], _config['_params_grid']), axis=1, result_type='expand')
    else:
        nodes_info["LONCOL"], nodes_info["LATCOL"] = tbd.GPS_to_grid(
            nodes_info["lng"], nodes_info["lat"], _config['_params_grid']
        )
        nodes_info["geometry"] = tbd.grid_to_polygon(
            [nodes_info["LONCOL"], nodes_info["LATCOL"]], _config['_params_grid']
        )
        nodes_info["lng_grid"], nodes_info["lat_grid"] = tbd.grid_to_centre(
            [nodes_info["LONCOL"], nodes_info["LATCOL"]], _config['_params_grid']
        )
        nodes_info["grid"] = (
            nodes_info["LONCOL"].astype(str) + "_" +
            nodes_info["LATCOL"].astype(str)
        )

    network_grid_nodes_info = gpd.GeoDataFrame()
    pool = mp.Pool(processes=mp.cpu_count())
    poolres = pool.map(partial(
        _merge_nodes_grid_, params=_config['_params_grid']), nodes_info.groupby(["grid"]))
    pool.close()

    cols = [col for col in nodes_info.columns if col not in ['geometry', 'lng_grid', 'lat_grid', 'loncol_1',
                                                             'loncol_2', 'loncol_3']]+['geometry', 'lng_grid', 'lat_grid', 'loncol_1', 'loncol_2', 'loncol_3', 'grid']
    for i, col_ in enumerate(cols):
        network_grid_nodes_info[col_] = [poolres[l][i]
                                         for l in range(len(poolres))]

    network_grid_nodes_info["x"] = network_grid_nodes_info["lng_grid"]
    network_grid_nodes_info["y"] = network_grid_nodes_info["lat_grid"]

    network_grid_nodes_info.set_index("grid", inplace=True)
    network_grid_nodes_info = network_grid_nodes_info.set_crs(
        settings.default_crs)

    logs.info("get network_grid_nodes_info successfully")

    ways_info = gpd.GeoDataFrame()
    ways_info = _ways.copy()

    ways_info[
        [
            "from_grid",
            "to_grid",
            "edgeid",
            "from_to",
            "edge_grid",
            "way_prjPt_geom",
            # "length_grid",
            # "hash_geometry",
            "ignore",
        ]
    ] = _ways.parallel_apply(  # _ways.parallel_apply(
        lambda way: waysinfo_update(
            way, nodes_info, _config['_params_grid']),
        axis=1,
        result_type="expand",
    )
    logs.info("get waysinfo_update-_ways.parallel_apply successfully")

    network_grid_edges_info = gpd.GeoDataFrame()
    pool = mp.Pool(processes=mp.cpu_count())
    poolres = pool.map(_merge_edges_grid_, ways_info.groupby(["from_to"]))
    pool.close()

    cols = [col for col in ways_info.columns if col not in ['geometry', 'way_prjPt_geom', 'length', "from_grid",
                                                            "to_grid", "from_to", "edgeid", "ignore"]]+['geometry', 'way_prjPt_geom', 'length', "from_grid",
                                                                                                        "to_grid", "from_to", "edgeid", "ignore", 'from_to']
    for i, col_ in enumerate(cols):
        network_grid_edges_info[col_] = [poolres[l][i]
                                         for l in range(len(poolres))]

    network_grid_edges_info.set_index("from_to", inplace=True)
    network_grid_edges_info = network_grid_edges_info.set_crs(
        settings.default_crs)

    """
    network_grid_edges_info = gpd.GeoDataFrame()
    for col in ways_info.columns:
        if col == "length_grid":
            network_grid_edges_info[col] = (
                ways_info.groupby(["from_to"])[col].apply(
                    lambda x: list(x)).to_frame()
            )
        elif col == "geometry":
            network_grid_edges_info[col] = (
                ways_info.groupby(["from_to"])[col].apply(
                    lambda x: MultiLineString(list(x))).to_frame()
            )
        elif col == "length":
            network_grid_edges_info[col] = (
                ways_info.groupby(["from_to"])[col].apply(
                    lambda x: max(list(x))).to_frame()
            )
        elif col in _ways.columns:
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
    """

    def merge_ignore_true(_ignore_false, ignore_true):
        if len(ignore_true[ignore_true['from_grid'] == _ignore_false['from_grid']]) > 0:
            _ignore_true = ignore_true[ignore_true['from_grid']
                                       == _ignore_false['from_grid']].iloc[0]
            for x in _ignore_true.index:
                if x not in ['geometry', 'from_grid', 'to_grid', 'edgeid', 'ignore']:
                    _ignore_false[x] = _ignore_true[x]+_ignore_false[x]
                if x == 'geometry':
                    _ignore_false[x] = MultiLineString([l for l in _ignore_true[x].geoms] +
                                                       [l for l in _ignore_false[x].geoms])
        if len(ignore_true[ignore_true['to_grid'] == _ignore_false['to_grid']]) > 0:
            _ignore_true = ignore_true[ignore_true['to_grid']
                                       == _ignore_false['to_grid']].iloc[0]
            for x in _ignore_true.index:
                if x not in ['geometry', 'from_grid', 'to_grid', 'edgeid', 'ignore']:
                    _ignore_false[x] = _ignore_false[x] + _ignore_true[x]
                if x == 'geometry':
                    _ignore_false[x] = MultiLineString([l for l in _ignore_false[x].geoms] +
                                                       [l for l in _ignore_true[x].geoms])
        return _ignore_false

    ignore_true = network_grid_edges_info[network_grid_edges_info["ignore"] == True].copy(
    )
    ignore_false = network_grid_edges_info[network_grid_edges_info["ignore"] == False].copy(
    )

    # network_grid_edges_info[network_grid_edges_info["ignore"] == False] = ignore_false.parallel_apply(
    #    lambda _ignore_false: merge_ignore_true(_ignore_false, ignore_true), axis=1)

    logs.info("get network_grid_edges_info successfully")

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
    if params_bound["method"] == 'hexa':
        # if len(node["grid"]) > 1:
        loncol_1, loncol_2, loncol_3 = tbd.GPS_to_grid(
            np.mean(node["lng_grid"]), np.mean(
                node["lat_grid"]), params_bound
        )

        lng_grid_new, lat_grid_new = tbd.grid_to_centre(
            [loncol_1[0], loncol_2[0], loncol_3[0]],
            params_bound,
        )
        geometry = tbd.grid_to_polygon(
            [loncol_1, loncol_2, loncol_3], params_bound
        )
        return loncol_1[0], loncol_2[0], loncol_3[0], str(loncol_1[0]) + "_" + str(loncol_2[0]) + "_" + str(loncol_3[0]), geometry[0], lng_grid_new[0], lat_grid_new[0]
        # else:
        # return node["loncol_1"][0], node["loncol_2"][0], node["loncol_3"][0], node["grid"][0], node["geometry"][0], node["lng_grid"][0], node["lat_grid"][0]
    else:
        if len(node["grid"]) > 1:
            LONCOL, LATCOL = tbd.GPS_to_grid(
                np.mean(node["lng_grid"]), np.mean(
                    node["lat_grid"]), params_bound
            )
            lng_grid_new, lat_grid_new = tbd.grid_to_centre(
                [LONCOL, LATCOL],
                params_bound,
            )
            geometry = tbd.grid_to_polygon(
                [LONCOL, LATCOL], params_bound
            )
            return LONCOL, LATCOL, str(LONCOL) + "_" + str(LATCOL), geometry, lng_grid_new, lat_grid_new
        else:
            return node["LONCOL"][0], node["LATCOL"][0], node["grid"][0], node["geometry"][0], node["lng_grid"][0], node["lat_grid"][0]


def _waysinfo_update(
    edge,
    # network_grid_cl_nodes_info,
    cl_grid_new,
    cl_grid_org,
    # params_bound,

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
    """
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
        geometry_cl, params_bound
    )

    length_grid = round(basicfunc.getlinestringlen(geometry_cl), 2)
    """
    return (
        from_grid_new,
        to_grid_new,
        from_grid_new + "-" + to_grid_new,
        (from_grid_new, to_grid_new),
        ignore_cl,
        # geometry_cl,
        # edge_grid_new,
        # length_grid,
    )


def _merge_nodes_cl_grid_(groupby_grid, params_method):
    group_name, group_data = groupby_grid
    if params_method == 'hexa':
        columns = ["loncol_1_cl", "loncol_2_cl", "loncol_3_cl",
                   "lng_grid_cl", "lat_grid_cl", "grid_cl"]
        rel = list()
        for col in group_data.columns:
            if col not in columns:
                rel.append(list(group_data[col]))

        rel.append(list(group_data['loncol_1_cl'])[0])
        rel.append(list(group_data['loncol_2_cl'])[0])
        rel.append(list(group_data['loncol_3_cl'])[0])
        rel.append(list(group_data['lng_grid_cl'])[0])
        rel.append(list(group_data['lat_grid_cl'])[0])
        rel.append(list(group_data['grid_cl'])[0])

    else:
        columns = ["LONCOL_cl", "LATCOL_cl",
                   "lng_grid_cl", "lat_grid_cl", "grid_cl"]
        rel = list()
        for col in group_data.columns:
            if col not in columns:
                rel.append(list(group_data[col]))

        rel.append(list(group_data['LONCOL_cl'])[0])
        rel.append(list(group_data['LATCOL_cl'])[0])
        rel.append(list(group_data['lng_grid_cl'])[0])
        rel.append(list(group_data['lat_grid_cl'])[0])
        rel.append(list(group_data['grid_cl'])[0])

    return tuple(_rel for _rel in rel)


def _merge_edges_cl_grid_(groupby_grid):
    group_name, group_data = groupby_grid

    columns = ["geometry", "way_prjPt_geom", "length", "from_grid_new",
               "to_grid_new", "from_to_new", "edgeid_new", "ignore_cl"]
    rel = list()
    for col in group_data.columns:
        if col not in columns:
            rel.append(list(group_data[col]))
    rel.append(MultiLineString(
        [l for ml in list(group_data['geometry']) for l in ml.geoms]))
    rel.append(MultiLineString(
        [l for ml in list(group_data['way_prjPt_geom']) for l in ml.geoms]))
    rel.append(group_data['length'].mean())
    rel.append(list(group_data['from_grid_new'])[0])
    rel.append(list(group_data['to_grid_new'])[0])
    rel.append(list(group_data['edgeid_new'])[0])
    rel.append(list(group_data['ignore_cl'])[0])
    rel.append(", ".join([str(i) for i in list(group_data['from_grid'])]))
    rel.append(", ".join([str(i) for i in list(group_data['to_grid'])]))
    rel.append(group_data['edge_grid'].sum())
    rel.append(group_name)
    return tuple(_rel for _rel in rel)


def get_network_cl_grid_info(
    network_grid_nodes_info,
    network_grid_edges_info,
    _config,
    mergenearnodes=True,
):
    buffer_mergenearnodes = _config["_buffer_mergenearnodes"]
    cl_nodes_info = gpd.GeoDataFrame()
    cl_nodes_info["grid"] = network_grid_nodes_info.index
    cl_nodes_info.set_index("grid", inplace=True)
    cl_nodes_info[["lng_grid", "lat_grid", "geometry", "loncol_1", "loncol_2", "loncol_3"]] = network_grid_nodes_info[
        ["lng_grid", "lat_grid", "geometry", "loncol_1", "loncol_2", "loncol_3"]
    ]
    if _config["_params_grid_cluster"]['method'] == 'hexa':
        def getgrid(_lng, _lat, _params):
            loncol_1_loncol_2_loncol_3_cl = tbd.GPS_to_grid(
                _lng, _lat, _params
            )
            loncol_1_cl, loncol_2_cl, loncol_3_cl = (loncol_cl[0]
                                                     for loncol_cl in loncol_1_loncol_2_loncol_3_cl)
            lng_grid_cl, lat_grid_cl = tbd.grid_to_centre(
                loncol_1_loncol_2_loncol_3_cl, _params)
            grid_cl = '_'.join([str(x[0])
                               for x in loncol_1_loncol_2_loncol_3_cl])
            return loncol_1_cl, loncol_2_cl, loncol_3_cl, lng_grid_cl[0], lat_grid_cl[0], grid_cl
        cl_nodes_info[['loncol_1_cl', 'loncol_2_cl', 'loncol_3_cl', 'lng_grid_cl', 'lat_grid_cl', 'grid_cl']] = cl_nodes_info.parallel_apply(
            lambda x: getgrid(x['lng_grid'], x['lat_grid'], _config['_params_grid_cluster']), axis=1, result_type='expand')

    else:
        cl_nodes_info[["LONCOL", "LATCOL"]
                      ] = network_grid_nodes_info[["LONCOL", "LATCOL"]]
        cl_nodes_info["LONCOL_cl"], cl_nodes_info["LATCOL_cl"] = tbd.GPS_to_grid(
            network_grid_nodes_info["lng_grid"],
            network_grid_nodes_info["lat_grid"],
            _config["_params_grid_cluster"],
        )
        cl_nodes_info["lng_grid_cl"], cl_nodes_info["lat_grid_cl"] = tbd.grid_to_centre(
            [cl_nodes_info["LONCOL_cl"], cl_nodes_info["LATCOL_cl"]
             ], _config["_params_grid_cluster"]
        )
        cl_nodes_info["grid_cl"] = (
            cl_nodes_info["LONCOL_cl"].astype(str)
            + "_"
            + cl_nodes_info["LATCOL_cl"].astype(str)
        )

    cl_edges_info = network_grid_edges_info[
        [
            "from_grid",
            "to_grid",
            "edgeid",
            "edge_grid",
            "geometry",
            "way_prjPt_geom",
            "length",
            "ignore",
        ]
    ].copy()
    cl_edges_info.reset_index(inplace=True)
    cl_edges_info.drop(columns='from_to', inplace=True)
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
    pool = mp.Pool(processes=mp.cpu_count())
    poolres = pool.map(partial(
        _merge_nodes_cl_grid_, params_method=_config['_params_grid']['method']),
        cl_nodes_info.reset_index().groupby(["mergenearnodes"]))
    pool.close()

    cols = [col for col in cl_nodes_info.reset_index().columns if col not in ["loncol_1_cl", "loncol_2_cl", "loncol_3_cl", "lng_grid_cl",
                                                                              "lat_grid_cl", "grid_cl"]]+["loncol_1_cl", "loncol_2_cl", "loncol_3_cl", "lng_grid_cl",
                                                                                                          "lat_grid_cl", "grid_cl"]
    for i, col_ in enumerate(cols):
        network_grid_cl_nodes_info[col_] = [poolres[l][i]
                                            for l in range(len(poolres))]

    network_grid_cl_nodes_info.drop(columns="mergenearnodes", inplace=True)

    if _config["_params_grid_cluster"]['method'] == 'hexa':
        network_grid_cl_nodes_info[
            ["loncol_1_new", "loncol_2_new", "loncol_3_new",
                "grid_new", "geometry", "lng_grid_new", "lat_grid_new"]
            # network_grid_cl_nodes_info.parallel_apply(
        ] = network_grid_cl_nodes_info.parallel_apply(
            lambda node: _nodesinfo_update(node, _config["_params_grid"]),
            axis=1,
            result_type="expand",
        )
    else:
        network_grid_cl_nodes_info[
            ["LONCOL_new", "LATCOL_new", "grid_new",
                "geometry", "lng_grid_new", "lat_grid_new"]
            # network_grid_cl_nodes_info.parallel_apply(
        ] = network_grid_cl_nodes_info.parallel_apply(
            lambda node: _nodesinfo_update(node, _config["_params_grid"]),
            axis=1,
            result_type="expand",
        )

    network_grid_cl_nodes_info["x"] = network_grid_cl_nodes_info["lng_grid_new"]
    network_grid_cl_nodes_info["y"] = network_grid_cl_nodes_info["lat_grid_new"]

    network_grid_cl_nodes_info.set_index("grid_new", inplace=True)
    network_grid_cl_nodes_info = network_grid_cl_nodes_info.set_crs(
        settings.default_crs)
    logs.info("get network_grid_cl_nodes_info successfully")

    cl_grid_new = network_grid_cl_nodes_info.index.tolist()
    cl_grid_org = network_grid_cl_nodes_info["grid"].tolist()
    cl_edges_info[
        [
            "from_grid_new",
            "to_grid_new",
            "edgeid_new",
            "from_to_new",
            "ignore_cl",
            # "geometry",
            # "edge_grid_new",
            # "length_grid",
        ]
    ] = cl_edges_info.parallel_apply(  # cl_edges_info.parallel_apply(
        lambda edge: _waysinfo_update(
            edge,
            # network_grid_cl_nodes_info,
            cl_grid_new,
            cl_grid_org,
            # _config["_params_grid"],
        ),
        axis=1,
        result_type="expand",
    )

    network_grid_cl_edges_info = gpd.GeoDataFrame()
    pool = mp.Pool(processes=mp.cpu_count())
    poolres = pool.map(_merge_edges_cl_grid_,
                       cl_edges_info.groupby("from_to_new"))
    pool.close()

    cols = [col for col in cl_edges_info.columns if col not in ["geometry", "way_prjPt_geom", "length", "from_grid_new", "to_grid_new", "edgeid_new",
                                                                "ignore_cl", "from_to_new"]]+["geometry", "way_prjPt_geom", "length", "from_grid_new",
                                                                                              "to_grid_new", "edgeid_new", "ignore_cl",
                                                                                              "from_grid_str", "to_grid_str", 'edge_grid_extend',
                                                                                              "from_to_new"]
    for i, col_ in enumerate(cols):
        network_grid_cl_edges_info[col_] = [poolres[l][i]
                                            for l in range(len(poolres))]

    network_grid_cl_edges_info.set_index("from_to_new", inplace=True)
    network_grid_cl_edges_info = network_grid_cl_edges_info.set_crs(
        settings.default_crs)

    def merge_ignore_true(_ignore_false, ignore_true):
        if len(ignore_true[ignore_true['from_grid_new'] == _ignore_false['from_grid_new']]) > 0:
            _ignore_true = ignore_true[ignore_true['from_grid_new']
                                       == _ignore_false['from_grid_new']].iloc[0]
            for x in _ignore_true.index:
                if x not in ['geometry', 'from_grid_new', 'to_grid_new', 'edgeid_new', 'ignore_cl']:
                    _ignore_false[x] = _ignore_true[x]+_ignore_false[x]
                if x == 'geometry':
                    _ignore_false[x] = MultiLineString([l for l in _ignore_true[x].geoms] +
                                                       [l for l in _ignore_false[x].geoms])
        if len(ignore_true[ignore_true['to_grid_new'] == _ignore_false['to_grid_new']]) > 0:
            _ignore_true = ignore_true[ignore_true['to_grid_new']
                                       == _ignore_false['to_grid_new']].iloc[0]
            for x in _ignore_true.index:
                if x not in ['geometry', 'from_grid_new', 'to_grid_new', 'edgeid_new', 'ignore_cl']:
                    _ignore_false[x] = _ignore_false[x] + _ignore_true[x]
                if x == 'geometry':
                    _ignore_false[x] = MultiLineString([l for l in _ignore_false[x].geoms] +
                                                       [l for l in _ignore_true[x].geoms])
        return _ignore_false

    # ignore_true = network_grid_cl_edges_info[network_grid_cl_edges_info["ignore_cl"] == True].copy(
    # )
    # ignore_false = network_grid_cl_edges_info[network_grid_cl_edges_info["ignore_cl"] == False].copy(
    # )

    # network_grid_cl_edges_info[network_grid_cl_edges_info["ignore_cl"] == False] = ignore_false.parallel_apply(
    #    lambda _ignore_false: merge_ignore_true(_ignore_false, ignore_true), axis=1)

    logs.info("get network_grid_cl_edges_info successfully")

    return (
        cl_nodes_info,
        network_grid_cl_nodes_info,
        cl_edges_info,
        network_grid_cl_edges_info,
    )


def edge_grid_to_gridid(edge_grid, grid_method):
    """
    eg.
    edge_grid=['3054_-514_-3568', '3056_-513_-3569', '3058_-512_-3570']
    return = [loncol_1,loncol_2,loncol_3] : Series
    """
    if grid_method == 'hexa':
        return [[int(s.split('_')[i]) for s in edge_grid] for i in range(3)]
    else:
        return [[int(s.split('_')[i]) for s in edge_grid] for i in range(2)]
