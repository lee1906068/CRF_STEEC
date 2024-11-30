import pandas as pd
import transbigdata as tbd
import geopandas as gpd
import numpy as np
import geopy.distance
import psycopg2
import pickle
import os
from shapely import unary_union, wkt, ops
from shapely.geometry import LineString, MultiLineString
import multiprocessing as mp
from functools import partial
from pandarallel import pandarallel

from . import log, settings, graph_from_geopandas_

"""
conn_info = {"conn_info_database": "tra_db",
             "conn_info_user": "postgres",
             "conn_info_password": "0000",
             "conn_info_host": "localhost",
             "conn_info_port": "5432"}

config = {
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
    "tofilepath":"prcdata/"
}


"""


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


class GridRoadNet:
    def __init__(self, analysisbounds=None, conn_info=None, config=None, loadexist=False, exisfile_root=None, _object=None, _object_pathset=None):
        if not loadexist:
            self.conn_info = conn_info
            self.config = config
            self.analysisbounds = analysisbounds

            self.areabounds = self.get_areabounds()
            self.params_grid = self.get_gridparams(self.areabounds,
                                                   self.config['grid_method'],
                                                   self.config['grid_accuracy'])  # type: ignore
            self.params_grid_cluster = self.get_gridparams(self.areabounds,
                                                           self.config['grid_method'],
                                                           self.config['grid_accuracy_cluster'])  # type: ignore
            self.nodes_all, \
                self.ways_all, \
                self.detectors_inarea = self.get_nodes_ways_dets(self.config['roadnet_nodes_db'],
                                                                 self.config['roadnet_edges_db'],
                                                                 self.config['det_roadnet_nodes_db'],
                                                                 self.config['det_roadnet_edges_db'],
                                                                 self.config['det_roads_db'],
                                                                 self.config['bounds_buffer'])  # type: ignore

            log("get_nodes_ways_dets successfully")
            self.nodes_info, \
                self.ways_info, \
                self.network_grid_nodes_info, \
                self.network_grid_edges_info,\
                self.network_grid_edges_info_ignore_true,\
                self.network_grid_edges_info_ignore_false = self.get_network_grid_info()  # type: ignore
            log("get_network_grid_info successfully")

            if self.config['mergenearnodes']:
                self.cl_nodes_info, \
                    self.cl_edges_info, \
                    self.network_grid_cl_nodes_info, \
                    self.network_grid_cl_edges_info,\
                    self.network_grid_cl_edges_info_ignore_true,\
                    self.network_grid_cl_edges_info_ignore_false = self.get_network_cl_grid_info()  # type: ignore

                self.network_grid_cl_nodes_gridid_map = self.get_gridid_map()
                log("get_network_cl_grid_info successfully")
            self.G = self.get_gridroatnet_graph()
        else:
            self.loadexisdata(exisfile_root, _object, _object_pathset)

    def loadexisdata(self, exisfile_root, _object, _object_pathset):
        if exisfile_root.endswith('/'):
            exisfile_root = exisfile_root[:-1]
        if _object is not None:
            exisfilepath = f"{exisfile_root}/{_object}/{_object_pathset}"
        else:
            exisfilepath = exisfile_root

        with open(f"{exisfile_root}/conn_info.pkl", 'rb') as file:
            self.conn_info = pickle.load(file)
        with open(f"{exisfile_root}/config_gridroadnet.pkl", 'rb') as file:
            self.config = pickle.load(file)
            if self.config['_object'] is None:
                self.config['_object'] = _object
            if self.config['_object_pathset'] is None:
                self.config['_object_pathset'] = _object_pathset

        self.areabounds = self.get_areabounds()
        self.params_grid = self.get_gridparams(self.areabounds,
                                               self.config['grid_method'],
                                               self.config['grid_accuracy'])  # type: ignore
        self.params_grid_cluster = self.get_gridparams(self.areabounds,
                                                       self.config['grid_method'],
                                                       self.config['grid_accuracy_cluster'])  # type: ignore

        if os.path.exists(f"{exisfilepath}/detectors_inarea.pkl"):
            with open(f"{exisfilepath}/detectors_inarea.pkl", 'rb') as file:
                self.detectors_inarea = pickle.load(file)
        if os.path.exists(f"{exisfilepath}/nodes_info.pkl"):
            with open(f"{exisfilepath}/nodes_info.pkl", 'rb') as file:
                self.nodes_info = pickle.load(file)
        if os.path.exists(f"{exisfilepath}/ways_info.pkl"):
            with open(f"{exisfilepath}/ways_info.pkl", 'rb') as file:
                self.ways_info = pickle.load(file)
        if os.path.exists(f"{exisfilepath}/network_grid_nodes_info.pkl"):
            with open(f"{exisfilepath}/network_grid_nodes_info.pkl", 'rb') as file:
                self.network_grid_nodes_info = pickle.load(file)
        if os.path.exists(f"{exisfilepath}/network_grid_edges_info.pkl"):
            with open(f"{exisfilepath}/network_grid_edges_info.pkl", 'rb') as file:
                self.network_grid_edges_info = pickle.load(file)
        if os.path.exists(f"{exisfilepath}/network_grid_edges_info_ignore_true.pkl"):
            with open(f"{exisfilepath}/network_grid_edges_info_ignore_true.pkl", 'rb') as file:
                self.network_grid_edges_info_ignore_true = pickle.load(file)
        if os.path.exists(f"{exisfilepath}/network_grid_edges_info_ignore_false.pkl"):
            with open(f"{exisfilepath}/network_grid_edges_info_ignore_false.pkl", 'rb') as file:
                self.network_grid_edges_info_ignore_false = pickle.load(file)
        if os.path.exists(f"{exisfilepath}/detectors_inarea.pkl"):
            with open(f"{exisfilepath}/detectors_inarea.pkl", 'rb') as file:
                self.detectors_inarea = pickle.load(file)
        if os.path.exists(f"{exisfilepath}/cl_nodes_info.pkl"):
            with open(f"{exisfilepath}/cl_nodes_info.pkl", 'rb') as file:
                self.cl_nodes_info = pickle.load(file)
        if os.path.exists(f"{exisfilepath}/cl_edges_info.pkl"):
            with open(f"{exisfilepath}/cl_edges_info.pkl", 'rb') as file:
                self.cl_edges_info = pickle.load(file)
        if os.path.exists(f"{exisfilepath}/network_grid_cl_nodes_info.pkl"):
            with open(f"{exisfilepath}/network_grid_cl_nodes_info.pkl", 'rb') as file:
                self.network_grid_cl_nodes_info = pickle.load(file)
        if os.path.exists(f"{exisfilepath}/network_grid_cl_edges_info.pkl"):
            with open(f"{exisfilepath}/network_grid_cl_edges_info.pkl", 'rb') as file:
                self.network_grid_cl_edges_info = pickle.load(file)
        if os.path.exists(f"{exisfilepath}/network_grid_cl_edges_info_ignore_true.pkl"):
            with open(f"{exisfilepath}/network_grid_cl_edges_info_ignore_true.pkl", 'rb') as file:
                self.network_grid_cl_edges_info_ignore_true = pickle.load(file)
        if os.path.exists(f"{exisfilepath}/network_grid_cl_edges_info_ignore_false.pkl"):
            with open(f"{exisfilepath}/network_grid_cl_edges_info_ignore_false.pkl", 'rb') as file:
                self.network_grid_cl_edges_info_ignore_false = pickle.load(
                    file)
        if os.path.exists(f"{exisfilepath}/gridroatnet_graph_.pkl"):
            with open(f"{exisfilepath}/gridroatnet_graph_.pkl", 'rb') as file:
                self.G = pickle.load(file)
        self.network_grid_cl_nodes_gridid_map = self.get_gridid_map()

    def get_db_connection(self):
        return psycopg2.connect(
            database=self.conn_info['conn_info_database'],
            user=self.conn_info['conn_info_user'],
            password=self.conn_info['conn_info_password'],
            host=self.conn_info['conn_info_host'],
            port=self.conn_info['conn_info_port'],
        )

    def get_areabounds(self):
        areabounds_gpd = gpd.read_file(self.config['shpfilepath'])
        xmin_path, ymin_path, xmax_path, ymax_path = areabounds_gpd.total_bounds
        return [xmin_path, ymin_path, xmax_path, ymax_path]

    def get_gridparams(self, areabounds, grid_method, accuracy):

        lng1, lat1, lng2, lat2 = areabounds
        if (lng1 > lng2) | (lat1 > lat2) | (abs(lat1) > 90) | (abs(lng1) > 180) | (
                abs(lat2) > 90) | (abs(lng2) > 180):
            raise Exception(
                'Bounds error. The input bounds should be in the order of [lon1,lat1,lon2,lat2]. (lon1,lat1) is the lower left corner and (lon2,lat2) is the upper right corner.')
        latStart = min(lat1, lat2)
        lonStart = min(lng1, lng2)
        deltaLon = accuracy * 360 / \
            (2 * np.pi * 6371004 * np.cos((lat1 + lat2) * np.pi / 360))
        deltaLat = accuracy * 360 / (2 * np.pi * 6371004)

        params_grid = {'slon': lonStart,
                       'slat': latStart,
                       'deltalon': deltaLon,
                       'deltalat': deltaLat,
                       'theta': 0,
                       'method': grid_method,
                       'gridsize': accuracy}

        return params_grid

    def get_nodes_ways_dets(self,
                            roadnet_nodes_db,
                            roadnet_edges_db,
                            det_roadnet_nodes_db,
                            det_roadnet_edges_db,
                            det_roads_db,
                            bounds_buffer):

        conn = self.get_db_connection()

        buffer_bound_min = [
            geopy.distance.distance(kilometers=bounds_buffer)
            .destination((self.analysisbounds[1], self.analysisbounds[0]), bearing=225)
            .longitude,
            geopy.distance.distance(kilometers=bounds_buffer)
            .destination((self.analysisbounds[1], self.analysisbounds[0]), bearing=225)
            .latitude,
        ]
        buffer_bound_max = [
            geopy.distance.distance(kilometers=bounds_buffer)
            .destination((self.analysisbounds[3], self.analysisbounds[2]), bearing=45)
            .longitude,
            geopy.distance.distance(kilometers=bounds_buffer)
            .destination((self.analysisbounds[3], self.analysisbounds[2]), bearing=45)
            .latitude,
        ]
        self.bufferbounds = [buffer_bound_min[0], buffer_bound_min[1],
                             buffer_bound_max[0], buffer_bound_max[1]]

        bound_polygon_str = f"POLYGON(({self.bufferbounds[0]} {self.bufferbounds[1]},\
                                        {self.bufferbounds[0]} {self.bufferbounds[3]},\
                                        {self.bufferbounds[2]} {self.bufferbounds[3]},\
                                        {self.bufferbounds[2]} {self.bufferbounds[1]},\
                                        {self.bufferbounds[0]} {self.bufferbounds[1]}))"
        st_geomfromtext_str = f"ST_GeomFromText('{bound_polygon_str}',4326)"
        st_contains_str = f"ST_Contains({st_geomfromtext_str},geometry)"

        sql1 = f"SELECT * FROM {roadnet_nodes_db} where {st_contains_str}"
        sql2 = f"SELECT * FROM {roadnet_edges_db} where {st_contains_str}"

        sql3 = f"SELECT * FROM {det_roadnet_nodes_db} where {st_contains_str}"
        sql4 = f"SELECT * FROM {det_roadnet_edges_db} where {st_contains_str}"
        sql5 = f"SELECT * FROM {det_roads_db} where {st_contains_str}"

        roadnodes = gpd.GeoDataFrame.from_postgis(
            sql1, conn, "geometry", crs="4326")
        roadways = gpd.GeoDataFrame.from_postgis(
            sql2, conn, "geometry", crs="4326")
        det_nodes = gpd.GeoDataFrame.from_postgis(
            sql3, conn, "geometry", crs="4326")
        det_ways = gpd.GeoDataFrame.from_postgis(
            sql4, conn, "geometry", crs="4326")
        detectors_inarea = gpd.GeoDataFrame.from_postgis(
            sql5, conn, "geometry", crs="4326"
        )

        detectors_inarea.set_index("name", inplace=True)
        detectors_inarea["point"] = gpd.points_from_xy(
            detectors_inarea.lng, detectors_inarea.lat)

        roadnodes["ref"] = ""
        nodes_all = pd.concat([roadnodes, det_nodes]).set_crs(4326)
        nodes_all.set_index("osmid", inplace=True)

        ways_all = pd.concat([roadways, det_ways]).set_crs(4326)
        ways_all["from_to"] = ways_all.parallel_apply(
            lambda way_: (way_["from"], way_["to"]), axis=1
        )
        ways_all.set_index("from_to", inplace=True)

        if self.params_grid['method'] == 'hexa':
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
                lambda detector: self._get_detetorgrids(detector["geometry"], self.params_grid), axis=1, result_type="expand",
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
                lambda detector: self._get_detetorgrids(detector["geometry"], self.params_grid), axis=1, result_type="expand",
            )

        return nodes_all, ways_all, detectors_inarea

    def get_network_grid_info(self):
        nodes_info = self.nodes_all.copy()
        ways_info = self.ways_all.copy()
        nodes_info.reset_index(inplace=True)
        ways_info.reset_index(inplace=True)
        ways_info.drop(columns="from_to", inplace=True)

        if self.config['grid_method'] == 'hexa':
            def getgrid(_lng, _lat, _params):
                loncol_1_loncol_2_loncol_3 = tbd.GPS_to_grid(
                    _lng, _lat, _params
                )
                loncol_1, loncol_2, loncol_3 = (loncol[0]
                                                for loncol in loncol_1_loncol_2_loncol_3)  # type: ignore
                lng_grid, lat_grid = tbd.grid_to_centre(
                    loncol_1_loncol_2_loncol_3, _params)  # type: ignore
                grid = '_'.join([str(x[0])
                                for x in loncol_1_loncol_2_loncol_3])  # type: ignore
                return loncol_1, loncol_2, loncol_3, lng_grid[0], lat_grid[0], grid

            nodes_info[['loncol_1', 'loncol_2', 'loncol_3', 'lng_grid', 'lat_grid', 'grid']] = nodes_info.parallel_apply(
                lambda x: getgrid(x['lng'], x['lat'], self.params_grid), axis=1, result_type='expand')
        else:
            nodes_info["LONCOL"], nodes_info["LATCOL"] = tbd.GPS_to_grid(
                nodes_info["lng"], nodes_info["lat"], self.params_grid
            )  # type: ignore
            nodes_info["geometry"] = tbd.grid_to_polygon(
                [nodes_info["LONCOL"], nodes_info["LATCOL"]], self.params_grid
            )  # type: ignore
            nodes_info["lng_grid"], nodes_info["lat_grid"] = tbd.grid_to_centre(
                [nodes_info["LONCOL"], nodes_info["LATCOL"]], self.params_grid
            )  # type: ignore
            nodes_info["grid"] = (
                nodes_info["LONCOL"].astype(str) + "_" +
                nodes_info["LATCOL"].astype(str)
            )

        network_grid_nodes_info = gpd.GeoDataFrame()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            poolres = pool.map(partial(
                self._merge_nodes_grid_, params=self.params_grid), nodes_info.groupby(["grid"]))
        # pool.close()
        # pool.join()

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

        log("get_network_grid_info_merge_nodes_grid_ successfully")

        ways_info[
            [
                "from_grid",
                "to_grid",
                "edgeid",
                "from_to",
                "edge_grid",
                "way_prjPt_geom",
                "ignore",
            ]
        ] = ways_info.parallel_apply(
            lambda way: self._waysinfo_update(
                way, nodes_info, self.params_grid),
            axis=1,
            result_type="expand",
        )
        log("get_network_grid_info_waysinfo_update successfully")

        network_grid_edges_info = gpd.GeoDataFrame()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            poolres = pool.map(self._merge_edges_grid_,
                               ways_info.groupby(["from_to"]))
        # pool.close()
        # pool.join()

        cols = [col for col in ways_info.columns if col not in ['geometry', 'way_prjPt_geom', 'length', "from_grid",
                                                                "to_grid", "from_to", "edgeid", "ignore"]]+['geometry', 'way_prjPt_geom', 'length', "from_grid",
                                                                                                            "to_grid", "from_to", "edgeid", "ignore", 'from_to']
        for i, col_ in enumerate(cols):
            network_grid_edges_info[col_] = [poolres[l][i]
                                             for l in range(len(poolres))]

        network_grid_edges_info.set_index("from_to", inplace=True)
        network_grid_edges_info = network_grid_edges_info.set_crs(
            settings.default_crs)
        log("get_network_grid_info_merge_edges_grid_ successfully")

        network_grid_edges_info_ignore_true = network_grid_edges_info[network_grid_edges_info["ignore"] == True].copy(
        )
        network_grid_edges_info_ignore_false = network_grid_edges_info[network_grid_edges_info["ignore"] == False].copy(
        )

        return nodes_info, \
            ways_info, \
            network_grid_nodes_info, \
            network_grid_edges_info, \
            network_grid_edges_info_ignore_true, \
            network_grid_edges_info_ignore_false

    def get_network_cl_grid_info(self):
        cl_nodes_info = gpd.GeoDataFrame()
        cl_nodes_info["grid"] = self.network_grid_nodes_info.index
        cl_nodes_info.set_index("grid", inplace=True)
        cl_nodes_info[["lng_grid", "lat_grid", "geometry", "loncol_1", "loncol_2", "loncol_3"]] = self.network_grid_nodes_info[
            ["lng_grid", "lat_grid", "geometry",
                "loncol_1", "loncol_2", "loncol_3"]
        ]
        if self.config['grid_method'] == 'hexa':
            def getgrid(_lng, _lat, _params):
                loncol_1_loncol_2_loncol_3_cl = tbd.GPS_to_grid(
                    _lng, _lat, _params
                )
                loncol_1_cl, loncol_2_cl, loncol_3_cl = (loncol_cl[0]
                                                         for loncol_cl in loncol_1_loncol_2_loncol_3_cl)  # type: ignore
                lng_grid_cl, lat_grid_cl = tbd.grid_to_centre(
                    loncol_1_loncol_2_loncol_3_cl, _params)  # type: ignore
                grid_cl = '_'.join([str(x[0])
                                    for x in loncol_1_loncol_2_loncol_3_cl])  # type: ignore
                return loncol_1_cl, loncol_2_cl, loncol_3_cl, lng_grid_cl[0], lat_grid_cl[0], grid_cl

            cl_nodes_info[['loncol_1_cl', 'loncol_2_cl', 'loncol_3_cl',
                           'lng_grid_cl', 'lat_grid_cl', 'grid_cl']] = cl_nodes_info.parallel_apply(
                lambda x: getgrid(x['lng_grid'], x['lat_grid'], self.params_grid_cluster), axis=1, result_type='expand')  # type: ignore
        else:
            cl_nodes_info[["LONCOL",
                           "LATCOL"]
                          ] = self.network_grid_nodes_info[["LONCOL", "LATCOL"]]  # type: ignore
            cl_nodes_info["LONCOL_cl"], \
                cl_nodes_info["LATCOL_cl"] = tbd.GPS_to_grid(
                self.network_grid_nodes_info["lng_grid"],
                self.network_grid_nodes_info["lat_grid"],
                self.params_grid_cluster,
            )  # type: ignore
            cl_nodes_info["lng_grid_cl"], cl_nodes_info["lat_grid_cl"] = tbd.grid_to_centre(
                [cl_nodes_info["LONCOL_cl"], cl_nodes_info["LATCOL_cl"]
                 ], self.params_grid_cluster
            )  # type: ignore
            cl_nodes_info["grid_cl"] = (
                cl_nodes_info["LONCOL_cl"].astype(str)
                + "_"
                + cl_nodes_info["LATCOL_cl"].astype(str)
            )
        log(
            "get_network_cl_grid_info_getgrid_cluster successfully")
        cl_edges_info = self.network_grid_edges_info[
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

        cl_nodes_info = self._mergenearnodes(
            cl_nodes_info, cl_edges_info, self.config['buffer_mergenearnodes']
        )
        log(
            "get_network_cl_grid_info_mergenearnodes successfully")

        network_grid_cl_nodes_info = gpd.GeoDataFrame()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            poolres = pool.map(partial(
                self._merge_nodes_cl_grid_, params_method=self.config['grid_method']),
                cl_nodes_info.reset_index().groupby(["mergenearnodes"]))
        # pool.close()
        # pool.join()

        cols = [col for col in cl_nodes_info.reset_index().columns if col not in ["loncol_1_cl", "loncol_2_cl", "loncol_3_cl", "lng_grid_cl",
                                                                                  "lat_grid_cl", "grid_cl"]]+["loncol_1_cl", "loncol_2_cl", "loncol_3_cl", "lng_grid_cl",
                                                                                                              "lat_grid_cl", "grid_cl"]
        for i, col_ in enumerate(cols):
            network_grid_cl_nodes_info[col_] = [poolres[l][i]
                                                for l in range(len(poolres))]

        network_grid_cl_nodes_info.drop(columns="mergenearnodes", inplace=True)
        log(
            "get_network_cl_grid_info_merge_nodes_cl_grid_ successfully")

        if self.config['grid_method'] == 'hexa':
            network_grid_cl_nodes_info[
                ["loncol_1_new", "loncol_2_new", "loncol_3_new",
                    "grid_new", "geometry", "lng_grid_new", "lat_grid_new"]
            ] = network_grid_cl_nodes_info.parallel_apply(
                lambda node: self._nodesinfo_update_cl(node, self.params_grid),
                axis=1,
                result_type="expand",
            )  # type: ignore
        else:
            network_grid_cl_nodes_info[
                ["LONCOL_new", "LATCOL_new", "grid_new",
                    "geometry", "lng_grid_new", "lat_grid_new"]
            ] = network_grid_cl_nodes_info.parallel_apply(
                lambda node: self._nodesinfo_update_cl(node, self.params_grid),
                axis=1,
                result_type="expand",
            )  # type: ignore

        network_grid_cl_nodes_info["x"] = network_grid_cl_nodes_info["lng_grid_new"]
        network_grid_cl_nodes_info["y"] = network_grid_cl_nodes_info["lat_grid_new"]

        network_grid_cl_nodes_info.set_index("grid_new", inplace=True)
        network_grid_cl_nodes_info = network_grid_cl_nodes_info.set_crs(
            settings.default_crs)
        log("get_network_cl_grid_info_nodesinfo_update_cl successfully")

        cl_grid_new = network_grid_cl_nodes_info.index.tolist()
        cl_grid_org = network_grid_cl_nodes_info["grid"].tolist()
        cl_edges_info[
            [
                "from_grid_new",
                "to_grid_new",
                "edgeid_new",
                "from_to_new",
                "ignore_cl",
            ]
        ] = cl_edges_info.parallel_apply(
            lambda edge: self._waysinfo_update_cl(
                edge,
                cl_grid_new,
                cl_grid_org,
            ),
            axis=1,
            result_type="expand",
        )  # type: ignore

        log(
            "get_network_cl_grid_info_waysinfo_update_cl successfully")

        network_grid_cl_edges_info = gpd.GeoDataFrame()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            poolres = pool.map(self._merge_edges_cl_grid_,
                               cl_edges_info.groupby("from_to_new"))
        # pool.close()
        # pool.join()

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
        log(
            "get_network_cl_grid_info_merge_edges_cl_grid_ successfully")

        network_grid_cl_edges_info_ignore_true = network_grid_cl_edges_info[
            network_grid_cl_edges_info["ignore_cl"] == True]
        network_grid_cl_edges_info_ignore_false = network_grid_cl_edges_info[
            network_grid_cl_edges_info["ignore_cl"] == False]

        return (
            cl_nodes_info,
            cl_edges_info,
            network_grid_cl_nodes_info,
            network_grid_cl_edges_info,
            network_grid_cl_edges_info_ignore_true,
            network_grid_cl_edges_info_ignore_false
        )

    def get_gridroatnet_graph(self):
        if self.config['mergenearnodes']:
            return graph_from_geopandas_(
                self.network_grid_cl_nodes_info,
                self.network_grid_cl_edges_info,
                "ignore_cl",
            )
        else:
            return graph_from_geopandas_(
                self.network_grid_nodes_info,
                self.network_grid_edges_info,
                "ignore",
            )

    def datatofile(self):
        _path = self.config['tofilepath']
        if self.config['tofilepath'].endswith('/'):
            _path = _path[:-1]
        _path = f"{_path}/gridroatnet/{self.config['_object']}/{self.config['_object_pathset']}"
        if not os.path.exists(_path):
            os.makedirs(_path)

        pickle.dump(self.detectors_inarea, open(
            f"{_path}/detectors_inarea.pkl", 'wb'))

        pickle.dump(self.G, open(f"{_path}/gridroatnet_graph_.pkl", 'wb'))
        pickle.dump(self.nodes_info, open(f"{_path}/nodes_info.pkl", 'wb'))
        pickle.dump(self.ways_info, open(f"{_path}/ways_info.pkl", 'wb'))
        pickle.dump(self.network_grid_nodes_info, open(
            f"{_path}/network_grid_nodes_info.pkl", 'wb'))
        pickle.dump(self.network_grid_edges_info, open(
            f"{_path}/network_grid_edges_info.pkl", 'wb'))
        pickle.dump(self.network_grid_edges_info_ignore_true, open(
            f"{_path}/network_grid_edges_info_ignore_true.pkl", 'wb'))
        pickle.dump(self.network_grid_edges_info_ignore_false, open(
            f"{_path}/network_grid_edges_info_ignore_false.pkl", 'wb'))
        if self.config['mergenearnodes']:
            pickle.dump(self.cl_nodes_info, open(
                f"{_path}/cl_nodes_info.pkl", 'wb'))
            pickle.dump(self.cl_edges_info, open(
                f"{_path}/cl_edges_info.pkl", 'wb'))
            pickle.dump(self.network_grid_cl_nodes_info, open(
                f"{_path}/network_grid_cl_nodes_info.pkl", 'wb'))
            pickle.dump(self.network_grid_cl_edges_info, open(
                f"{_path}/network_grid_cl_edges_info.pkl", 'wb'))
            pickle.dump(self.network_grid_cl_edges_info_ignore_true, open(
                f"{_path}/network_grid_cl_edges_info_ignore_true.pkl", 'wb'))
            pickle.dump(self.network_grid_cl_edges_info_ignore_false, open(
                f"{_path}/network_grid_cl_edges_info_ignore_false.pkl", 'wb'))

    def get_gridid_map(self):
        _grid_list = self.network_grid_cl_nodes_info['grid'].values.tolist()
        _grid_new_list = self.network_grid_cl_nodes_info.index.tolist()

        grid_list = self.flatten(_grid_list)
        grid_new_list = self.flatten([[grid_new] * len(_grid_list[i])
                                      for i, grid_new in enumerate(_grid_new_list)])

        network_grid_cl_nodes_gridid_map = pd.DataFrame(
            {"grid": grid_list, "grid_new": grid_new_list}).set_index("grid")
        return network_grid_cl_nodes_gridid_map

    def _get_detetorgrids(self, _detector, params_grid):
        bounds = _detector.bounds
        deltaLon = params_grid["deltalon"]
        deltaLat = params_grid["deltalat"]
        tmppoints = pd.DataFrame(
            np.array(
                np.meshgrid(
                    np.arange(bounds[0],
                              bounds[2]+deltaLon/3,
                              deltaLon/3),
                    np.arange(bounds[1],
                              bounds[3]+deltaLat/3,
                              deltaLat/3))
            ).reshape(2, -1).T, columns=['lon', 'lat'])  # type: ignore
        if params_grid['method'] == 'hexa':
            tmppoints['loncol_1'], tmppoints['loncol_2'], tmppoints['loncol_3'] = tbd.GPS_to_grid(
                tmppoints['lon'], tmppoints['lat'], params_grid)  # type: ignore
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
            tmppoints['LONCOL'], tmppoints['LATCOL'] = tbd.GPS_to_grid(
                tmppoints['lon'], tmppoints['lat'], params_grid)  # type: ignore
            tmppoints = tmppoints[['LONCOL', 'loncol_3']].drop_duplicates()
            tmppoints["grid"] = tmppoints["LONCOL"].astype(
                str) + "_" + tmppoints["loncol_3"].astype(str)
            return tmppoints["grid"].tolist(), \
                tmppoints[["LONCOL", "LATCOL"]].values.tolist(),\
                tmppoints["LONCOL"].min(), tmppoints["LONCOL"].max(),\
                tmppoints["LATCOL"].min(), tmppoints["LATCOL"].max()

    def _traj_to_grids(self, traj, params):
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

    def _network_edge_grid(self, way_geometry, params_grid):
        if way_geometry.length > 0:
            edge_grid = self._traj_to_grids(gpd.GeoDataFrame([way_geometry],
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

    def _waysinfo_update(self, way, nodes_info, params_bound):
        from_grid = nodes_info[nodes_info["osmid"]
                               == way["from"]]["grid"].values[0]
        to_grid = nodes_info[nodes_info["osmid"]
                             == way["to"]]["grid"].values[0]
        ignore = True if from_grid == to_grid else False

        edge_grid, way_prjPt_geom = self._network_edge_grid(
            way["geometry"], params_bound
        )

        return (
            from_grid,
            to_grid,
            from_grid + "-" + to_grid,
            (from_grid, to_grid),
            edge_grid,
            way_prjPt_geom,
            ignore,
        )

    def _merge_nodes_grid_(self, groupby_grid, params):
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

    def _merge_edges_grid_(self, groupby_grid):
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
    """
    def _mergenearnodes(self, cl_nodes_info, cl_edges_info, buffer_mergenearnodes=60):
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

        cl_nodes_info["mergenearnodes"] = pd.concat(
            [
                pd.DataFrame.from_dict(_grid_cl_list, orient="index")
                for _grid_cl_list in grid_cl_list
            ]
        )
        return cl_nodes_info
    """

    def _mergenearnodes_thread(self, _cl, cl_nodes_info, cl_edges_info, buffer_mergenearnodes=60):
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
        return grid_cl_dict

    def _mergenearnodes(self, cl_nodes_info, cl_edges_info, buffer_mergenearnodes=60):
        grid_cl = set(cl_nodes_info["grid_cl"].values.tolist())
        # grid_cl_list = list()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            grid_cl_list = pool.map(partial(
                self._mergenearnodes_thread, cl_nodes_info=cl_nodes_info, cl_edges_info=cl_edges_info, buffer_mergenearnodes=buffer_mergenearnodes),
                grid_cl)
        # pool.close()
        # pool.join()

        cl_nodes_info["mergenearnodes"] = pd.concat(
            [
                pd.DataFrame.from_dict(_grid_cl_list, orient="index")
                for _grid_cl_list in grid_cl_list
            ]
        )
        return cl_nodes_info

    def _nodesinfo_update_cl(self, node, params_bound):
        if params_bound["method"] == 'hexa':
            loncol_1, loncol_2, loncol_3 = tbd.GPS_to_grid(
                np.mean(node["lng_grid"]), np.mean(node["lat_grid"]),
                params_bound
            )  # type: ignore

            lng_grid_new, lat_grid_new = tbd.grid_to_centre(
                [loncol_1[0], loncol_2[0], loncol_3[0]],
                params_bound,
            )  # type: ignore
            geometry = tbd.grid_to_polygon(
                [loncol_1, loncol_2, loncol_3], params_bound
            )
            return loncol_1[0], loncol_2[0], loncol_3[0], str(loncol_1[0]) + "_" + str(loncol_2[0]) + "_" + str(loncol_3[0]), geometry[0], lng_grid_new[0], lat_grid_new[0]
        else:
            if len(node["grid"]) > 1:
                LONCOL, LATCOL = tbd.GPS_to_grid(
                    np.mean(node["lng_grid"]), np.mean(
                        node["lat_grid"]), params_bound
                )  # type: ignore
                lng_grid_new, lat_grid_new = tbd.grid_to_centre(
                    [LONCOL, LATCOL],
                    params_bound,
                )  # type: ignore
                geometry = tbd.grid_to_polygon(
                    [LONCOL, LATCOL], params_bound
                )
                return LONCOL, LATCOL, str(LONCOL) + "_" + str(LATCOL), geometry, lng_grid_new, lat_grid_new
            else:
                return node["LONCOL"][0], node["LATCOL"][0], node["grid"][0], node["geometry"][0], node["lng_grid"][0], node["lat_grid"][0]

    def _waysinfo_update_cl(self, edge, cl_grid_new, cl_grid_org):
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

        return (
            from_grid_new,
            to_grid_new,
            from_grid_new + "-" + to_grid_new,
            (from_grid_new, to_grid_new),
            ignore_cl)  # type: ignore

    def _merge_nodes_cl_grid_(self, groupby_grid, params_method):
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

    def _merge_edges_cl_grid_(self, groupby_grid):
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

    def flatten(self, li):
        return sum(([x] if not isinstance(x, list) else self.flatten(x) for x in li), [])
