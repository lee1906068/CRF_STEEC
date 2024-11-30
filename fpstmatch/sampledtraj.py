import pandas as pd
import transbigdata as tbd
import geopandas as gpd
import numpy as np
from datetime import timedelta
from itertools import product
import pickle
import os

from shapely import unary_union, wkt, ops
from shapely.geometry import LineString, MultiLineString, GeometryCollection
import multiprocessing as mp
from functools import partial
from pandarallel import pandarallel

from . import log, settings, gridroadnet, distance, basicfunc

"""
conn_info = {"conn_info_database": "tra_db",
                "conn_info_user": "postgres",
                "conn_info_password": "0000",
                "conn_info_host": "localhost",
                "conn_info_port": "5432"}

config_gridroadnet = {
    "shpfilepath": 'bj_boundary/bj_boundary.shp',
    "roadnet_nodes_db": "osm_bj_nodes",
    "roadnet_edges_db": "osm_bj_edges",
    "det_roadnet_nodes_db": "detectorscandidateroads_nodes",
    "det_roadnet_edges_db": "detectorscandidateroads_edges",
    "det_roads_db": "detectorscandidateroads",
    "bounds_buffer": 0.5,  # kilometers
    "grid_method": "hexa",
    "grid_accuracy": 25,  # meters
    "mergenearnodes": True,
    "grid_accuracy_cluster": 25 * 2,  # meters
    "buffer_mergenearnodes": 25 * 2,
    "_object": _object_pathset[0],
    "_object_pathset": _object_pathset[1],
    "tofilepath": "prcdata/"
}

config_strprc = {
    "delta_T": timedelta(minutes=30),
    "delta_L": 2000,
    "K": 20,
    "strsegment":True,
    "strseg":strseg,#### if "strsegment":False
    "gridroadnet_loadexist":False,
    "exisfile_root":'/home/lee/Documents/PycharmProjects/cityRoute/prcdata/gridroatnet',#### if "gridroadnet_loadexist":True
    "_object": _object_pathset[0],
    "_object_pathset": _object_pathset[1],
    "get_strseglps":True,
    "tofilepath": "prcdata/"
}

"""


class SampledTraj:
    def __init__(self, rawstr=None, config_strprc=None, conn_info=None, config_gridroadnet=None) -> None:
        self.rawstr = rawstr
        self.config_strprc = config_strprc
        if self.config_strprc['strsegment']:
            self.strseg = self.strsegment(
                self.rawstr, self.config_strprc['delta_T'], self.config_strprc['delta_L'])
        else:
            self.strseg = self.config_strprc['strseg'].copy()

        if not self.config_strprc['gridroadnet_loadexist']:
            self.gRN = gridroadnet.GridRoadNet(analysisbounds=self.rawstr.total_bounds,
                                               conn_info=conn_info, config=config_gridroadnet)
        else:
            self.gRN = gridroadnet.GridRoadNet(loadexist=True, exisfile_root=self.config_strprc['exisfile_root'],
                                               _object=self.config_strprc['_object'], _object_pathset=self.config_strprc['_object_pathset'])
        if self.config_strprc['get_strseglps']:
            self.lps_var = self.get_strseglps()

    def strsegment(self, rawstr, delta_T, delta_L) -> pd.DataFrame:
        segstr = pd.DataFrame()
        segstr["object"] = rawstr["id_object"]
        segstr["pathset"] = rawstr["id_pathset"]
        segstr["det_name_o"] = rawstr["name"].shift()
        segstr["det_name_d"] = rawstr["name"]
        segstr["det_area_o"] = rawstr["geometry"].shift()
        segstr["det_area_d"] = rawstr["geometry"]
        segstr["lat_o"] = rawstr["lat"].shift()
        segstr["lng_o"] = rawstr["lng"].shift()
        segstr["lat_d"] = rawstr["lat"]
        segstr["lng_d"] = rawstr["lng"]
        segstr["time_o"] = rawstr["time"].shift()
        segstr["time_d"] = rawstr["time"]
        segstr["tSeg"] = (
            ((rawstr["time"] - rawstr["time"].shift())
             > delta_T).astype(int).cumsum()
        )
        segstr["lSeg"] = (
            (
                tbd.getdistance(
                    rawstr["lng"],
                    rawstr["lat"],
                    rawstr["lng"].shift(),
                    rawstr["lat"].shift(),
                )
                > delta_L
            )
            .astype(int)
            .cumsum()
        )

        for ts, subtr in segstr.groupby("tSeg"):
            segstr.loc[segstr[segstr["tSeg"].isin([ts])].index, "lSeg"] = (
                subtr["lSeg"] - subtr["lSeg"].min()
            )
        segstr["tlSeg"] = segstr["tSeg"].astype(
            str) + "_" + segstr["lSeg"].astype(str)

        segstr["dist_nsp"] = tbd.getdistance(
            segstr["lng_o"],
            segstr["lat_o"],
            segstr["lng_d"],
            segstr["lat_d"],
        )
        segstr["dur_nsp"] = segstr["time_d"] - segstr["time_o"]
        segstr["dur_nsp_seconds"] = segstr.parallel_apply(
            lambda _data: _data.dur_nsp.seconds, axis=1)  # type: ignore
        for ts, subtr in segstr.groupby("tlSeg"):
            segstr.drop(subtr.head(1).index, inplace=True)
        segstr.reset_index(inplace=True)

        return segstr

    def get_strseglps(self):
        self.strseg[["odpairs", "odpaths", "K"]] = self.strseg.parallel_apply(
            lambda seg: self._getlpset_(seg, K=self.config_strprc['K']),
            axis=1,
            result_type="expand")  # type: ignore

        return self.get_lps_var()

    def get_lps_var(self):
        lps_var_dict = dict()
        for idx_tl, _tlseg in self.strseg.groupby('tlSeg'):
            self.detectors_inarea = self.gRN.detectors_inarea[
                self.gRN.detectors_inarea["type"].isin(["camera"])
            ]
            op = _tlseg.head(1).iloc[0]['det_area_o'].centroid
            dp = _tlseg.tail(1).iloc[0]['det_area_d'].centroid
            self.o_point = [op.x, op.y]
            self.d_point = [dp.x, dp.y]
            self.od_head = basicfunc.LatLng2Degree(
                self.o_point[1],
                self.o_point[0],
                self.d_point[1],
                self.d_point[0],
            )
            _speed_len_dirh_dirt_detc_detd_detl_ = list()
            for idx, self.seg in _tlseg.iterrows():
                if 'ignore' in self.seg['odpaths']:
                    continue
                ways = self.seg['odpaths']
                with mp.Pool(processes=mp.cpu_count()) as pool:
                    poolres = pool.map(self._get_var_len_dir_det_, ways)
                _speed_len_dirh_dirt_detc_detd_detl_.append(
                    np.array(poolres))

            lps_var_dict[idx_tl] = pad_arrays(
                _speed_len_dirh_dirt_detc_detd_detl_)
        return lps_var_dict

    def datatofile(self):
        _path = self.config_strprc['tofilepath']
        if self.config_strprc['tofilepath'].endswith('/'):
            _path = _path[:-1]
        _path = f"{_path}/sampledtraj_{self.config_strprc['K']}/{self.config_strprc['_object']}/{self.config_strprc['_object_pathset']}"
        if not os.path.exists(_path):
            os.makedirs(_path)

        pickle.dump(self.strseg, open(
            f"{_path}/{self.config_strprc['_object']}_{self.config_strprc['_object_pathset']}_strseg.pkl", 'wb'))
        for ltseg in self.lps_var:
            np.save(
                f"{_path}/{self.config_strprc['_object']}_{self.config_strprc['_object_pathset']}_{ltseg}_speed_len_dirh_dirt_detc_detd_detl.npy", self.lps_var[ltseg])

    def _getlpset_(self, seg, K=10):
        det_grids_o = self.gRN._traj_to_grids(
            seg['det_area_o'].bounds, self.gRN.params_grid)
        det_grids_o["grid"] = (
            det_grids_o["loncol_1"].astype(str) + "_" +
            det_grids_o["loncol_2"].astype(str) + "_" +
            det_grids_o["loncol_3"].astype(str)
        )
        det_grids_d = self.gRN._traj_to_grids(
            seg['det_area_d'].bounds, self.gRN.params_grid)
        det_grids_d["grid"] = (
            det_grids_d["loncol_1"].astype(str) + "_" +
            det_grids_d["loncol_2"].astype(str) + "_" +
            det_grids_d["loncol_3"].astype(str)
        )

        mask_o = self.gRN.network_grid_cl_nodes_gridid_map.index.isin(
            det_grids_o["grid"].values)
        nodes_in_det_grids_o = list(
            set(flatten(self.gRN.network_grid_cl_nodes_gridid_map.loc[mask_o].values.tolist())))

        mask_d = self.gRN.network_grid_cl_nodes_gridid_map.index.isin(
            det_grids_d["grid"].values)
        nodes_in_det_grids_d = list(set(flatten(
            self.gRN.network_grid_cl_nodes_gridid_map.loc[mask_d].values.tolist())))
        odpairs = list()
        for _temp in list(product(nodes_in_det_grids_o, nodes_in_det_grids_d)):
            if _temp[0] != _temp[1]:
                odpairs.append(_temp)

        shortest_paths = list()
        if len(odpairs) == 0:
            return "ignore", "ignore", 0

        for _odpairs in odpairs:
            from_, to_ = _odpairs
            try:
                _shortest_paths = list(
                    distance.k_shortest_paths(
                        self.gRN.G, from_, to_, K, weight="length")
                )
                shortest_paths.extend(_shortest_paths)
            except:
                continue
        if len(shortest_paths) == 0:
            return odpairs, "ignore", 0
        else:
            return odpairs, shortest_paths, len(shortest_paths)

    def _edge_grid_union(self, edge_grid_list):
        if len(edge_grid_list) > 1:
            l = set(edge_grid_list[0])
            for t in edge_grid_list[1:]:
                l = l.union(t)
        else:
            l = set(edge_grid_list[0])

        return gridroadnet.edge_grid_to_gridid(flatten(list(l)), "hexa")

    def to_grid_info_exp(self, sub_route, ignore_true):
        insert_grid = ignore_true[ignore_true['from_grid_str'].str.contains(
            "|".join(sub_route['to_grid']))]
        if len(insert_grid) == 0:
            geometry_exp = sub_route['geometry']
            way_prjPt_geom_exp = sub_route['way_prjPt_geom']
            length_exp = sub_route['length']
            edge_grid_exp = sub_route['edge_grid_extend']
            edge_grid_union = self._edge_grid_union(edge_grid_exp)
        else:
            geometry_exp = GeometryCollection(
                [sub_route['geometry']]+list(insert_grid['geometry']))
            way_prjPt_geom_exp = GeometryCollection(
                [sub_route['way_prjPt_geom']]+list(insert_grid['way_prjPt_geom']))
            length_exp = sub_route['length']+insert_grid['length'].sum()
            edge_grid_exp = sub_route['edge_grid_extend'] + \
                insert_grid['edge_grid_extend'].sum()
            edge_grid_union = self._edge_grid_union(edge_grid_exp)

        return geometry_exp, way_prjPt_geom_exp, length_exp, edge_grid_exp, edge_grid_union

    def _get_var_len_dir_det_(self, _ways):
        lines_list = list()
        for u, v in zip(_ways[:-1], _ways[1:]):
            lines_list.append(list(self.gRN.G.get_edge_data(u, v).values())[0])
        lines = gpd.GeoDataFrame(pd.DataFrame(lines_list))
        lines[['geometry_exp', 'way_prjPt_geom_exp', 'length_exp', 'edge_grid_exp', 'edge_grid_union']] = lines.apply(
            lambda sub_route: self.to_grid_info_exp(sub_route, self.gRN.network_grid_cl_edges_info_ignore_true), axis=1, result_type='expand')

        _speed, _delta_len = self._get_var_len(self.seg, lines)
        _dir_head, _dir_turn = self._get_var_dir(
            self.d_point, self.od_head, lines)
        _det_count, _det_dist, _det_len = self._get_var_det(
            self.seg, lines)

        return _speed, _delta_len, _dir_head, _dir_turn, _det_count, _det_dist, _det_len

    def _get_var_len(self, seg, lines):
        path_delta_len_norm = 1 - abs(lines['length_exp'].sum()-seg['dist_nsp']) / \
            max(lines['length_exp'].sum(), seg['dist_nsp'])
        return round(lines['length_exp'].sum()/seg['dur_nsp_seconds'], 2), round(path_delta_len_norm, 2)

    def _get_var_dir(self, d_point, od_head, lines):
        h = list()
        t = list()
        for idx, line in lines.iterrows():
            # break
            _h = list()
            _t = list()
            for _line in line['way_prjPt_geom'].geoms:
                _h.append(abs(od_head-np.mean([basicfunc.LatLng2Degree(
                    list(_line.coords)[i][1],
                    list(_line.coords)[i][0],
                    d_point[1],
                    d_point[0],
                ) for i in range(len(list(_line.coords)))])))

                _t.append(abs(od_head-np.mean([basicfunc.LatLng2Degree(
                    list(_line.coords)[i][1],
                    list(_line.coords)[i][0],
                    list(_line.coords)[i+1][1],
                    list(_line.coords)[i+1][0],
                ) for i in range(len(list(_line.coords))-1)])))
            h.append(np.mean(_h))
            t.append(np.mean(_t))

        path_dir_head_norm = 1-abs(np.mean(h)-180) / 180 \
            if np.mean(h) > 180 else 1 - np.mean(h)/180
        path_dir_turn_norm = 1-abs(np.mean(t)-180) / 180 \
            if np.mean(t) > 180 else 1 - np.mean(t)/180

        return np.round(path_dir_head_norm, 2), np.round(path_dir_turn_norm, 2)

    def _get_var_det(self, seg, lines):
        _det_count_ = list()
        _det_dist_ = list()
        _det_len_ = list()
        for idx, line in lines.iterrows():
            tra_grids_loncol_1_min = np.min(line['edge_grid_union'][0])
            tra_grids_loncol_1_max = np.max(line['edge_grid_union'][0])
            tra_grids_loncol_2_min = np.min(line['edge_grid_union'][1])
            tra_grids_loncol_2_max = np.max(line['edge_grid_union'][1])
            tra_grids_loncol_3_min = np.min(line['edge_grid_union'][2])
            tra_grids_loncol_3_max = np.max(line['edge_grid_union'][2])
            detectors = self.detectors_inarea.drop(
                index=list(set(self.detectors_inarea.index.tolist()).intersection(set([seg["det_name_o"], seg["det_name_d"]])))).copy()
            _detectors = detectors.query(
                f"((loncol_1_min <={tra_grids_loncol_1_max}) and (loncol_1_max >={tra_grids_loncol_1_min})) and (loncol_2_min <={tra_grids_loncol_2_max}) and ((loncol_2_max >={tra_grids_loncol_2_min})) and ((loncol_3_min <={tra_grids_loncol_3_max}) and (loncol_3_max >={tra_grids_loncol_3_min}))"
            )
            _detectors_ = _detectors.set_geometry(
                "point", crs=4326).to_crs(3395)
            if len(_detectors_) > 0:
                for detname, det in _detectors_.iterrows():
                    inter = set(det["grids"]).intersection(
                        set(flatten(line['edge_grid'])))
                    if len(inter) > 0:
                        _det_count_.append(detname)
                        _det_dist_.append(len(inter) / len(det["grids"]))
                        _det_len_.append(np.mean([gpd.GeoDataFrame([{"geometry": _line}], crs=4326).to_crs(
                            3395)["geometry"].values[0].distance(det["point"]) for _line in line['geometry'].geoms]))
        det_count = len(_det_count_)
        det_dist = np.nan_to_num(np.sum(_det_dist_))
        det_len = np.nan_to_num(np.sum(_det_len_))
        path_det_count_norm = np.exp(-det_count)
        path_det_dist_norm = 1-det_dist / det_count if det_count != 0 else 1
        path_det_len_norm = det_len / det_count/50 if det_count != 0 else 1
        if path_det_len_norm > 1:
            path_det_len_norm = 1

        return np.round(path_det_count_norm, 2), np.round(path_det_dist_norm, 2), np.round(path_det_len_norm, 2)


def flatten(li):
    return sum(([x] if not isinstance(x, list) else flatten(x) for x in li), [])


def pad_arrays(arrays):
    # 获取三个数组中最大的第一维长度
    max_length = max([a.shape[0] for a in arrays])

    # 构造新数组
    padded_arrays = np.zeros((len(arrays), max_length, arrays[0].shape[1]))

    # 对每个数组进行 pad
    for i, a in enumerate(arrays):
        pad_length = max_length - a.shape[0]
        pad_width = ((0, pad_length), (0, 0))
        padded_arrays[i] = np.pad(
            a, pad_width=pad_width, mode='constant', constant_values=-1)

    return padded_arrays
