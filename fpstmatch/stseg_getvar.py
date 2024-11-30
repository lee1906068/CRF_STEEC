import os
import random
import time

import geopandas as gpd
import numpy as np
import pandas as pd

from . import creategraph, distance, logs, pe_func_vars, stats, util_coord, utils_graph


def _get_basicvar_(
    _object,
    _pathset,
    i_tlsegs,
    index,
    K_max,
    dist_nsp,
    dur_nsp,
    det_name_o,
    det_name_d,
    odpaths,
    network_nodes_info,
    network_edges_info,
    detectors_inarea,
    bound_grid,
    from_name,
    to_name,
):
    if not os.path.exists(f"prcdata/{_object}/{_pathset}/lp_path/{i_tlsegs}/{index}"):
        os.makedirs(f"prcdata/{_object}/{_pathset}/lp_path/{i_tlsegs}/{index}")

    if "ignore" in odpaths:
        return None

    _lp_paths = list()
    _lp_len = list()
    _lp_speed = list()
    _lp_dir_head = list()
    _lp_dir_turn = list()
    _lp_det_count = list()
    _lp_det_dist = list()
    _lp_det_len = list()
    detectors = detectors_inarea.drop(index=[det_name_o, det_name_d])
    
    for k, odpath in enumerate(odpaths):
        
        (
            lp_paths,
            lp_len,
            lp_speed,
            lp_dir_head,
            lp_dir_turn,
            lp_det_count,
            lp_det_dist,
            lp_det_len,
        ) = pe_func_vars.pe_basicvars(
            network_nodes_info,
            network_edges_info,
            odpath,
            dist_nsp,
            dur_nsp,
            detectors,
            bound_grid,
            from_name,
            to_name,
        )

        lp_paths.to_csv(f"prcdata/{_object}/{_pathset}/lp_path/{i_tlsegs}/{index}/{k}.csv")

        _lp_len.append(lp_len)
        _lp_speed.append(lp_speed)
        _lp_dir_head.append(lp_dir_head)
        _lp_dir_turn.append(lp_dir_turn)
        _lp_det_count.append(lp_det_count)
        _lp_det_dist.append(lp_det_dist)
        _lp_det_len.append(lp_det_len)

    _lp_len_np = np.pad(
        _lp_len, (0, K_max - len(_lp_len)), "constant", constant_values=-1
    )
    _lp_speed_np = np.pad(
        _lp_speed, (0, K_max - len(_lp_speed)), "constant", constant_values=-1
    )
    _lp_dir_head_np = np.pad(
        _lp_dir_head, (0, K_max - len(_lp_dir_head)), "constant", constant_values=-1
    )
    _lp_dir_turn_np = np.pad(
        _lp_dir_turn, (0, K_max - len(_lp_dir_turn)), "constant", constant_values=-1
    )
    _lp_det_count_np = np.pad(
        _lp_det_count, (0, K_max - len(_lp_det_count)), "constant", constant_values=-1
    )
    _lp_det_dist_np = np.pad(
        _lp_det_dist, (0, K_max - len(_lp_det_dist)), "constant", constant_values=-1
    )
    _lp_det_len_np = np.pad(
        _lp_det_len, (0, K_max - len(_lp_det_len)), "constant", constant_values=-1
    )
    _lp_path_basicvar = np.stack(
        [
            _lp_len_np,
            _lp_speed_np,
            _lp_dir_head_np,
            _lp_dir_turn_np,
            _lp_det_count_np,
            _lp_det_dist_np,
            _lp_det_len_np,
        ],
        axis=1,
    )

    np.savetxt(
        f"prcdata/{_object}/{_pathset}/lp_path_basicvar/{i_tlsegs}/{index}.csv",
        _lp_path_basicvar,
        delimiter=",",
        fmt="%f",
    )
    return _lp_path_basicvar


def getstseg_pe_vars(ST_seg, config):

    start = time.time()
    from_name = config["gridnet_edge_from_name"]
    to_name = config["gridnet_edge_to_name"]
    params_bound = config["params_bound"]
    bound_grid = config["bound_grid"].copy()
    network_grid_cl_nodes_info = config["network_grid_cl_nodes_info"].copy()
    network_grid_cl_edges_info = config["network_grid_cl_edges_info"].copy()
    network_grid_cl_ignore = config["network_grid_cl_ignore"]
    
    detectors_inarea = config["detectors_inarea"].copy()   
    detectors_inarea["geometry"]=gpd.points_from_xy(detectors_inarea.lng, detectors_inarea.lat)
    detectors_inarea=detectors_inarea.to_crs(3395)

    bound_grid.set_index("grid",inplace=True)
    

    K_max = ST_seg["K"].max()
    lp_path_basicvar_list = list()
    st_eulid_dur_nsp_list = list()
    for i_tlsegs, tlsegs in ST_seg.groupby("tlSeg"):

        #logs.info(
        #    f"getstseg_pe_vars for {i_tlsegs} tlsegs of all {len(ST_seg.groupby('tlSeg'))} tlsegs"
        #)
        if not os.path.exists(f"prcdata/{ST_seg.iloc[0]['object']}/{ST_seg.iloc[0]['pathset']}/lp_path_basicvar/npy"):
            os.makedirs(f"prcdata/{ST_seg.iloc[0]['object']}/{ST_seg.iloc[0]['pathset']}/lp_path_basicvar/npy")
        if not os.path.exists(f"prcdata/{ST_seg.iloc[0]['object']}/{ST_seg.iloc[0]['pathset']}/lp_path_basicvar/{i_tlsegs}"):
            os.makedirs(f"prcdata/{ST_seg.iloc[0]['object']}/{ST_seg.iloc[0]['pathset']}/lp_path_basicvar/{i_tlsegs}")

        tlsegs["index"] = list(range(1, len(tlsegs) + 1))
        _lp_path_basicvar = tlsegs.parallel_apply(
            lambda x: _get_basicvar_(
                ST_seg.iloc[0]["object"],
                ST_seg.iloc[0]["pathset"],
                i_tlsegs,
                x["index"],
                K_max,
                x["dist_nsp"],
                x["dur_nsp"].seconds,
                x["det_name_o"],
                x["det_name_d"],
                x["odpaths"],
                network_grid_cl_nodes_info,
                network_grid_cl_edges_info,
                detectors_inarea,
                bound_grid,
                from_name,
                to_name,
            ),
            axis=1,
        )
        _st_eulid_dur_nsp = tlsegs.parallel_apply(
            lambda x: np.stack([x["dist_nsp"], x["dur_nsp"].seconds], axis=0)
            #if (not (len(x["nodes_o"])==1 and x["nodes_o"] == x["nodes_d"])) and x["time_o"]!=x["time_d"]
            if "ignore" not in x["odpaths"]
            else None,
            axis=1,
        )
        
        temp = [_basicvar for _basicvar in _lp_path_basicvar if _basicvar is not None]
        if len(temp)==0: continue
            
        lp_path_basicvar = np.stack(
            temp,
            axis=0,
        )

        st_eulid_dur_nsp = np.stack(
            [
                _st_eulid_dur
                for _st_eulid_dur in _st_eulid_dur_nsp
                if _st_eulid_dur is not None
            ],
            axis=0,
        )

        np.save(f"prcdata/{ST_seg.iloc[0]['object']}/{ST_seg.iloc[0]['pathset']}/lp_path_basicvar/npy/{i_tlsegs}.npy", lp_path_basicvar)
        np.save(
            f"prcdata/{ST_seg.iloc[0]['object']}/{ST_seg.iloc[0]['pathset']}/lp_path_basicvar/npy/{i_tlsegs}_eulid_dur.npy", st_eulid_dur_nsp
        )

        lp_path_basicvar_list.append(lp_path_basicvar)
        st_eulid_dur_nsp_list.append(st_eulid_dur_nsp)

    end = time.time()
    #logs.info(f"getstseg_pe_vars finish")
    #logs.info("Task_getstseg_pe_vars runs %0.2f seconds." % (end - start))
    return lp_path_basicvar_list, st_eulid_dur_nsp_list