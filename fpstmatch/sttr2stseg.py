import os
import time
from itertools import product

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import transbigdata as tbd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from . import creategraph, distance, logs, stats, util_coord, utils_graph, simplificationroadnet


def _getnodesofSP(STtr, sql_db_name, sql_conn, nodes_G):
    """
    get nodes of sample points in road networks

    """
    _nodes = list()
    for index, sampleposition in STtr.iterrows():
        sql_candidatenodes = (
            f"SELECT * FROM {sql_db_name} where ref like '{sampleposition['name']}-%'"
        )
        candidatenodes = gpd.GeoDataFrame.from_postgis(
            sql_candidatenodes, sql_conn, "geometry"
        )
        nodes_ = list()
        for candidatenode in candidatenodes["osmid"]:

            filter_data = nodes_G[nodes_G.index.astype(str).str.contains(
                str(candidatenode))]
            if len(filter_data) > 0:
                nodes_.extend(list(filter_data.index))
        _nodes.append(list(set(nodes_)))

    return _nodes


def _getnodesofSP_grid(
    STtr, sql_db_name, sql_conn, params_bound, cl_grid_new, cl_grid_org
):
    """
    get nodes of sample points grids in road networks

    """

    def qurey_cl_grid_new_cl_grid_org(qurey, grid_new, grid_org):
        for _ in zip(grid_new, grid_org):
            if qurey in _[1]:
                return _[0]

    _nodes = list()
    for index, sampleposition in STtr.iterrows():
        sql_candidatenodes = (
            f"SELECT * FROM {sql_db_name} where ref like '{sampleposition['name']}-%'"
        )
        candidatenodes = gpd.GeoDataFrame.from_postgis(
            sql_candidatenodes, sql_conn, "geometry"
        )
        LONCOL, LATCOL = tbd.GPS_to_grid(
            candidatenodes["lng"], candidatenodes["lat"], params_bound
        )
        if len(candidatenodes) > 1:
            _nodes.append(
                list(
                    {
                        qurey_cl_grid_new_cl_grid_org(
                            qurey, cl_grid_new, cl_grid_org)
                        for qurey in list(
                            {
                                str(LONCOL_LATCOL[0]) +
                                "_" + str(LONCOL_LATCOL[1])
                                for LONCOL_LATCOL in zip(LONCOL, LATCOL)
                            }
                        )
                    }
                )
            )
        elif len(candidatenodes) == 1:
            _nodes.append(
                list(
                    {
                        qurey_cl_grid_new_cl_grid_org(
                            qurey, cl_grid_new, cl_grid_org)
                        for qurey in list(
                            {
                                str(LONCOL_LATCOL[0]) +
                                "_" + str(LONCOL_LATCOL[1])
                                for LONCOL_LATCOL in zip([LONCOL], [LATCOL])
                            }
                        )
                    }
                )
            )
        else:
            _nodes.append(None)
    return _nodes


def _getnodesofSP_(STtr, detector_nodes):

    return STtr.parallel_apply(
        lambda x: list(
            set(
                detector_nodes[detector_nodes["ref"].str.contains(x["name"] + "-")][
                    "osmid"
                ].tolist()
            )
        ),
        axis=1,
    ).tolist()


def _divideST_TL(STtr, nodes, delta_T, delta_L):
    """
    divide ST to ST_seg with delta.T and delta.L

    """

    ST_seg = pd.DataFrame()

    ST_seg["id"] = STtr["index"]
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

    ST_seg.drop(ST_seg.head(1).index, inplace=True)

    """
    get Len_eulid between the nodes of ST_seg (neibor sample points)

    """

    ST_seg["dist_nsp"] = tbd.getdistance(
        ST_seg["lng_o"],
        ST_seg["lat_o"],
        ST_seg["lng_d"],
        ST_seg["lat_d"],
    )

    ST_seg["dur_nsp"] = ST_seg["time_d"] - ST_seg["time_o"]
    ST_seg["nodes_o"] = nodes[0: len(nodes) - 1]
    ST_seg["nodes_d"] = nodes[1: len(nodes)]

    startflag = True
    for ts, subtr in ST_seg.groupby("tSeg"):
        if startflag:
            startflag = False
            continue
        else:
            ST_seg.drop(subtr.head(1).index, inplace=True)

    return ST_seg


def _getodpairsofST(ST_seg, nodes):
    """
    get odpairs between the nodes of ST_seg (neibor sample points)

    """
    _ST_seg = ST_seg.copy()

    _odpairs = list()
    startflag = True
    for i, tlsegs in _ST_seg.groupby("tlSeg"):
        tlsegs.reset_index(inplace=True)
        for ii, tlseg in tlsegs.iterrows():
            note = ""
            if not startflag and ii == 0:
                note = "ignore_seg"
            if tlseg["det_name_o"] == tlseg["det_name_d"]:
                note = "ignore_stay" if note == "" else note + ";ignore_stay"
            if ST_seg["time_d"] == ST_seg["time_o"]:
                note = "ignore_ab" if note == "" else note + ";ignore_ab"
            if note != "":
                _odpairs.append(note)
                continue
            temp = list()
            for _temp in list(product(tlseg["nodes_o"], tlseg["nodes_d"])):
                if _temp[0] != _temp[1]:
                    temp.append(_temp)
            _odpairs.append(temp)
        startflag = False

    _ST_seg["odpairs"] = pd.DataFrame(
        {"odpairs": _odpairs}, index=_ST_seg.index)

    _ST_seg.drop(
        _ST_seg[_ST_seg["odpairs"].str.contains(
            "ignore_seg").isin([True])].index,
        inplace=True,
    )

    return _ST_seg


def _getlpset(ST_seg, roadsnetworks, K=10, bar1=None, bar2=None, bar3=None):
    """
    get lp of ST_seg (neibor sample points)

    """
    _ST_seg = ST_seg.copy()

    shortest_paths = dict()
    shortest_paths_K = dict()
    if bar1 is not None:
        bar1.reset(total=len(_ST_seg.groupby("tlSeg")))
    for i, tlsegs in _ST_seg.groupby("tlSeg"):
        if bar1 is not None:
            bar1.update()
        if bar2 is not None:
            bar2.reset(total=len(tlsegs))
        for ii, tlseg in tlsegs.iterrows():
            if bar2 is not None:
                bar2.update()
            shortest_paths[ii] = list()
            _k = 0
            if "ignore" in tlseg["odpairs"]:
                shortest_paths[ii] = tlseg["odpairs"]
                shortest_paths_K[ii] = int(_k * K)
                continue
            if bar3 is not None:
                bar3.reset(total=len(tlseg["odpairs"]))
            for iii, _odpairs in enumerate(tlseg["odpairs"]):
                if bar3 is not None:
                    bar3.update()
                from_, to_ = _odpairs
                _shortest_paths = list(
                    distance.k_shortest_paths(
                        roadsnetworks, from_, to_, K, weight="length"
                    )
                )
                shortest_paths[ii].extend(_shortest_paths)
                _k += 1
            shortest_paths_K[ii] = int(_k * K)
    _ST_seg["odpaths"] = pd.DataFrame(
        {"odpaths": shortest_paths}, index=_ST_seg.index)
    _ST_seg["K"] = pd.DataFrame({"K": shortest_paths_K}, index=_ST_seg.index)
    return _ST_seg


def _getodpairs_(det_name_o, det_name_d, nodes_o, nodes_d, time_o, time_d):
    if nodes_o == None or nodes_d == None:
        return "ignore"
    if det_name_o == det_name_d or time_o == time_d:
        return "ignore_stay"

    temp = list()
    for _temp in list(product(nodes_o, nodes_d)):
        if _temp[0] != _temp[1]:
            temp.append(_temp)
    if len(temp) == 0:
        return "ignore_stay"
    else:
        return temp


def _getlpset_(roadsnetworks, odpairs, K=10):
    shortest_paths = list()
    if "ignore" in odpairs:
        return odpairs, 0

    for iii, _odpairs in enumerate(odpairs):
        from_, to_ = _odpairs
        try:
            _shortest_paths = list(
                distance.k_shortest_paths(
                    roadsnetworks, from_, to_, K, weight="length")
            )
            shortest_paths.extend(_shortest_paths)
        except:
            continue
    if len(shortest_paths) == 0:
        return "ignore", 0
    else:
        return shortest_paths, len(shortest_paths)


def _getlpset_para(ST_seg, roadsnetworks, K=10):
    """
    get odpairs between the nodes of ST_seg (neibor sample points)
    get lp of ST_seg (neibor sample points)

    """
    _ST_seg = ST_seg.copy()
    _ST_seg["odpairs"] = _ST_seg.parallel_apply(
        lambda x: _getodpairs_(
            x["det_name_o"],
            x["det_name_d"],
            x["nodes_o"],
            x["nodes_d"],
            x["time_o"],
            x["time_d"],
        ),
        axis=1,
    )
    _ST_seg["odpaths"] = _ST_seg.parallel_apply(
        lambda x: _getlpset_(roadsnetworks, x["odpairs"], K)[0], axis=1
    )
    _ST_seg["K"] = _ST_seg.parallel_apply(
        lambda x: _getlpset_(roadsnetworks, x["odpairs"], K)[1], axis=1
    )
    return _ST_seg


def getSTseg(
    STtr,
    sql_conn,
    _config,
):
    start = time.time()

    sql_db_name = _config["sql_db_name"]
    delta_T = _config["delta_T"]
    delta_L = _config["delta_L"]
    K = _config["K"]
    nodes_ = _config["nodes_"]
    ways_ = _config["ways_"]
    gridflag = _config["gridflag"]
    para = _config["para"]

    if not gridflag:
        G = simplificationroadnet.createroadnetwithdetectors(
            nodes_, ways_)
        _nodes_ = nodes_.copy()
        _nodes_.set_index("osmid", inplace=True)
        if not para:
            _nodes = _getnodesofSP(
                STtr=STtr,
                sql_db_name=sql_db_name,
                sql_conn=sql_conn,
                nodes_G=_nodes_,
            )
            _ST_seg = _divideST_TL(
                STtr=STtr, nodes=_nodes, delta_T=delta_T, delta_L=delta_L
            )
            ST_seg_ = _getodpairsofST(ST_seg=_ST_seg, nodes=_nodes)
            ST_seg = _getlpset(
                ST_seg=ST_seg_,
                roadsnetworks=G,
                K=K,
                bar1=None,
                bar2=None,
                bar3=None,
            )
        else:
            # logs.info(f"get nodes of sample points in road networks")

            _nodes = _getnodesofSP(
                STtr=STtr,
                sql_db_name=sql_db_name,
                sql_conn=sql_conn,
                nodes_G=_nodes_,
            )
            # logs.info(f"divide ST to ST_seg with delta.T and delta.L")
            _ST_seg = _divideST_TL(
                STtr=STtr, nodes=_nodes, delta_T=delta_T, delta_L=delta_L
            )
            # logs.info(
            #    f"get odpairs and lp between the nodes of ST_seg (neibor sample points)"
            # )
            ST_seg = _getlpset_para(ST_seg=_ST_seg, roadsnetworks=G, K=K)
            # logs.info(f"getSTseg finish")
    else:
        network_grid_nodes_info = _config["network_grid_nodes_info"]
        network_grid_edges_info = _config["network_grid_edges_info"]
        network_grid_cl_nodes_info = _config["network_grid_cl_nodes_info"]
        network_grid_cl_edges_info = _config["network_grid_cl_edges_info"]
        network_grid_cl_ignore = _config["network_grid_cl_ignore"]
        cl_grid_new = network_grid_cl_nodes_info.index.tolist()
        cl_grid_org = network_grid_cl_nodes_info["grid"].tolist()
        params_bound = _config["params_bound"]

        G = creategraph.graph_from_geopandas_(
            network_grid_cl_nodes_info,
            network_grid_cl_edges_info,
            network_grid_cl_ignore,
        )
        # logs.info(f"get nodes of sample points grids in road networks")
        _grid_nodes = _getnodesofSP_grid(
            STtr=STtr,
            sql_db_name=sql_db_name,
            sql_conn=sql_conn,
            params_bound=params_bound,
            cl_grid_new=cl_grid_new,
            cl_grid_org=cl_grid_org,
        )
        # logs.info(f"divide ST to ST_seg with delta.T and delta.L")
        _ST_seg = _divideST_TL(
            STtr=STtr, nodes=_grid_nodes, delta_T=delta_T, delta_L=delta_L
        )
        # logs.info(
        #    f"get odpairs and lp between the nodes of ST_seg (neibor sample points)"
        # )
        ST_seg = _getlpset_para(ST_seg=_ST_seg, roadsnetworks=G, K=K)
        # logs.info(f"getSTseg finish")

    if not os.path.exists(f"prcdata/{ST_seg.iloc[0]['object']}/{ST_seg.iloc[0]['pathset']}"):
        os.makedirs(
            f"prcdata/{ST_seg.iloc[0]['object']}/{ST_seg.iloc[0]['pathset']}")
    ST_seg.to_csv(
        f"prcdata/{ST_seg.iloc[0]['object']}/{ST_seg.iloc[0]['pathset']}/ST_seg.csv")

    end = time.time()
    # logs.info("Task_getSTseg runs %0.2f seconds." % (end - start))
    return ST_seg
