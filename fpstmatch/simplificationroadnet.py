import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
import transbigdata as tbd
from . import creategraph
from . import utils_graph
from . import util_coord
from . import stats
from . import logs

def createroadnetwithdetectors(nodes_gdf, ways_gdf, trans_type='walk'):
    def concat_func(x):
        return pd.Series(
            {
                "id": x["id"].unique()[0],
                "geometry": x["geometry"].unique()[0],
                #'osmid':x['osmid'],
                "lat": x["lat"].unique()[0],
                "lng": x["lng"].unique()[0],
                "street_count": x["street_count"].unique()[0],
                "ref": (",".join(x["ref"].unique())),
                "highway": x["highway"].unique()[0],
            }
        )


    nodes_gdf = nodes_gdf.groupby(nodes_gdf["osmid"]).apply(concat_func).reset_index()
    nodes_gdf.reset_index(inplace=True)
    ways_gdf.reset_index(inplace=True)
    nodes_gdf["geometry_xy"] = None
    coord_array = np.array(nodes_gdf[["lng", "lat"]].values.tolist())
    central_lon, central_lat = np.mean(coord_array, axis=0)
    central_lon, central_lat = float(central_lon), float(central_lat)
    northern = True if central_lat >= 0 else False

    for index, node in nodes_gdf.iterrows():
        xs, ys = util_coord.from_latlon(node.lng, node.lat, central_lon)
        nodes_gdf.loc[index, "geometry_xy"] = Point(xs, ys)

    nodes_gdf["geometry_xy"] = None
    coord_array = np.array(nodes_gdf[["lng", "lat"]].values.tolist())
    central_lon, central_lat = np.mean(coord_array, axis=0)
    central_lon, central_lat = float(central_lon), float(central_lat)
    northern = True if central_lat >= 0 else False

    for index, node in nodes_gdf.iterrows():
        xs, ys = util_coord.from_latlon(node.lng, node.lat, central_lon)
        nodes_gdf.loc[index, "geometry_xy"] = Point(xs, ys)

    G = creategraph.graph_from_(
        nodes_gdf,
        ways_gdf,
        retain_all=True,
        network_type=trans_type,
        great_circle_distance=True,
    )
    logs.info(f"creategraph from roadnodes and roadedges complete")

    null_nodes = [n for n, sc in G.nodes(data="street_count") if sc is None]
    street_count = stats.count_streets_per_node(G, nodes=null_nodes)
    nx.set_node_attributes(G, street_count, name="street_count")

    logs.info(f"set_node_attributes with crossroads count complete")

    G.remove_nodes_from(
        utils_graph.graph_to_gdfs(G, edges=False)[
            utils_graph.graph_to_gdfs(G, edges=False)["street_count"].isin([0])
        ].index.to_list()
    )

    logs.info(f"remove_nodes_from for no crossroad nodes complete")
    logs.info(f"createroadnetwithdetectors complete")

    return G

def node_way2grid(_G, nodes_gdf, gridaccuracy=150,gridmethod='hexa', gridoptmethod='centerdist', printgridlog=False):
    G = _G.copy()
    
    _grid_rec, _params_rec = tbd.area_to_grid(nodes_gdf, method=gridmethod, accuracy=gridaccuracy)

    params_optimized = tbd.grid_params_optimize(
        nodes_gdf,
        _params_rec,
        col=["id", "lng", "lat"],
        optmethod=gridoptmethod,
        sample=0,  # not sampling
        printlog=printgridlog,
    )
    grid_rec, params_rec = tbd.area_to_grid(
        nodes_gdf, params=params_optimized, method="hexa", accuracy=gridaccuracy
    )
    logs.info(f"put area to grid complete")

    grid_rec[["loncol_1", "loncol_2", "loncol_3"]] = grid_rec[
        ["loncol_1", "loncol_2", "loncol_3"]
    ].astype(str)

    nodes, ways = utils_graph.graph_to_gdfs(_G)
    nodeswithgrid = gpd.sjoin(nodes, grid_rec, how="left", predicate="within")
    nodeswithgrid.rename(columns={"index_right": "cluster"}, inplace=True)
    nodeswithgrid["intersection_id"] = None
    nodeswithgrid["cluster"] = (
        nodeswithgrid["loncol_1"].astype(str)
        + "_"
        + nodeswithgrid["loncol_2"].astype(str)
        + "_"
        + nodeswithgrid["loncol_3"].astype(str)
    )
    nodes_groups = nodeswithgrid.groupby(["cluster"])

    wayswithgrid = gpd.sjoin(ways, grid_rec, how="left", predicate="intersects")
    wayswithgrid.rename(columns={"index_right": "cluster"}, inplace=True)
    wayswithgrid["cluster"] = (
        wayswithgrid["loncol_1"].astype(str)
        + "_"
        + wayswithgrid["loncol_2"].astype(str)
        + "_"
        + wayswithgrid["loncol_3"].astype(str)
    )

    G.remove_nodes_from(nodeswithgrid[nodeswithgrid["cluster"].isin(['nan_nan_nan'])].index.tolist())

    nodeswithgrid.drop(index=nodeswithgrid[nodeswithgrid["cluster"].isin(['nan_nan_nan'])].index, inplace=True)
    wayswithgrid.drop(index=wayswithgrid[wayswithgrid["cluster"].isin(['nan_nan_nan'])].index, inplace=True)

    ways_groups = wayswithgrid.groupby(["cluster"])

    logs.info(f"node_way2grid complete")

    return G, nodeswithgrid, wayswithgrid, grid_rec, params_rec


def nodescluster(nodeswithgrid, wayswithgrid, buffer=25):
    ways_groups = wayswithgrid.groupby(["cluster"])
    for ways_cl, ways_subset in ways_groups:
        group_list = list()
        group_status = list()
        for way_id, way in ways_subset.iterrows():
            if way["length"] > buffer:
                continue
            """
            try:
                if (
                    "camera" in nodeswithgrid.loc[way["from"]]["ref"]
                    or "wifi" in nodeswithgrid.loc[way["from"]]["ref"]
                ):
                    nodeswithgrid.loc[way["from"], "intersection_id"] = ways_cl + "_detector_"+nodeswithgrid.loc[way["from"]]["ref"]
                    continue
                if (
                    "camera" in nodeswithgrid.loc[way["to"]]["ref"]
                    or "wifi" in nodeswithgrid.loc[way["to"]]["ref"]
                ):
                    nodeswithgrid.loc[way["to"], "intersection_id"] = ways_cl + "_detector_"+nodeswithgrid.loc[way["to"]]["ref"]
                    continue
            except:
                continue
            """
            try:
                if not (
                    nodeswithgrid.loc[way["from"]]["intersection_id"] is None
                    and nodeswithgrid.loc[way["to"]]["intersection_id"] is None
                ):
                    continue
            except:
                continue

            group_list.append({way["from"], way["to"]})
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
                if (
                    node
                    in nodeswithgrid[
                        nodeswithgrid["cluster"].isin([ways_cl])
                    ].index.to_list()
                ):
                    nodeswithgrid.loc[node, "intersection_id"] = (
                        ways_cl + "_" + str(max_intersection_id)
                    )
                    flag = True
            if flag:
                max_intersection_id += 1
                flag = False

    logs.info(f"intersection road cluster complete")

    for index, isolate_node in nodeswithgrid[
        nodeswithgrid["intersection_id"].isin([None])
    ].iterrows():
        nodeswithgrid.loc[isolate_node["osmid"], "intersection_id"] = (
            isolate_node["cluster"] + "_" + str(index) + "_is"
        )

    node_group_dict = {}

    for node_id, node in nodeswithgrid.iterrows():
        if node.intersection_id in node_group_dict.keys():
            node_group_dict[node.intersection_id].append(node)
        else:
            node_group_dict[node.intersection_id] = [node]

    logs.info(f"nodescluster complete")
    return node_group_dict


def clusterRoadnet(
    _G,
    node_group_dict,
    nodeswithgrid,
    lonlat_coord_precision=7,
    local_coord_precision=2,
):
    G = _G.copy()
    H = nx.MultiDiGraph()
    H.graph = G.graph

    _nodes = list()
    for intersection_id, node_group in node_group_dict.items():
        new_node = dict()
        if intersection_id[len(intersection_id)-2:len(intersection_id)] != "is":
            new_node["intersection_id"] = intersection_id
            osm_node_id_list = []
            ref_node_id_list = []
            highway_node_id_list = []
            x_coord_sum, y_coord_sum = 0.0, 0.0
            x_coord_xy_sum, y_coord_xy_sum = 0.0, 0.0

            for node in node_group:
                osm_node_id_list.append(node.osmid)
                if node.ref is not None:
                    ref_node_id_list.append(node.ref)
                if node.highway is not None:
                    highway_node_id_list.append(node.highway)
                x_coord_sum += node.geometry.x
                y_coord_sum += node.geometry.y
                x_coord_xy_sum += node.geometry_xy.x
                y_coord_xy_sum += node.geometry_xy.y

            new_node["osmid_original"] = ",".join(
                list(map(lambda x: str(x), osm_node_id_list))
            )
            x_coord_ave = round(x_coord_sum/len(node_group), lonlat_coord_precision)
            y_coord_ave = round(y_coord_sum/len(node_group), lonlat_coord_precision)
            new_node["geometry"] = Point(x_coord_ave, y_coord_ave)
            x_coord_xy_ave = round(
                x_coord_xy_sum / len(node_group), local_coord_precision
            )
            y_coord_xy_ave = round(
                y_coord_xy_sum / len(node_group), local_coord_precision
            )
            new_node["geometry_xy"] = Point(x_coord_xy_ave, y_coord_xy_ave)

            new_node["x"] = x_coord_ave
            new_node["y"] = y_coord_ave
            new_node["osmid"] = intersection_id
            new_node["ref"] = (
                "".join(ref_node_id_list) if "".join(ref_node_id_list) != "" else None
            )
            new_node["highway"] = (
                ",".join(highway_node_id_list)
                if ",".join(highway_node_id_list) != ""
                else None
            )
            _nodes.append(new_node["osmid"])
            H.add_node(new_node["osmid"], **new_node)
        else:
            for node_id, node in enumerate(node_group):
                new_node["intersection_id"] = intersection_id
                new_node["osmid_original"] = str(node.osmid)
                new_node["geometry"] = node.geometry
                new_node["geometry_xy"] = node.geometry_xy
                new_node["x"] = node.x
                new_node["y"] = node.y
                new_node["osmid"] = intersection_id
                new_node["ref"] = node.ref
                new_node["highway"] = node.highway

                _nodes.append(new_node["osmid"])
                H.add_node(new_node["osmid"], **new_node)

    logs.info(f"new_node form clusternode add to graph complete")
    edge_count = 0
    for u, v, k, data in G.edges(keys=True, data=True):

        u2 = nodeswithgrid.loc[u, "intersection_id"]
        v2 = nodeswithgrid.loc[v, "intersection_id"]

        # only create the edge if we're not connecting the cluster
        # to itself, but always add original self-loops
        if u2 not in _nodes:
            print(u2)
        if (u2 != v2) or (u == v):
            data["from_original"] = u
            data["to_original"] = v
            data["from"] = u2
            data["to"] = v2
            H.add_edge(u2, v2, **data)
        edge_count += 1

    groups_ = nodeswithgrid.groupby(["intersection_id"])
    for cluster_label, nodes_subset in groups_:
        # but only if there were multiple nodes merged together,
        # otherwise it's the same old edge as in original graph
        if len(nodes_subset) > 1:
            # get coords of merged nodes point centroid to prepend or
            # append to the old edge geom's coords

            # for each edge incident on this new merged node, update its
            # geometry to extend to/from the new node's point coords
            in_edges = set(H.in_edges(cluster_label, keys=True))
            out_edges = set(H.out_edges(cluster_label, keys=True))
            for u, v, k in in_edges | out_edges:
                x = H.nodes[cluster_label]["x"]
                y = H.nodes[cluster_label]["y"]
                xy = [(x, y)]
                old_coords = list(H.edges[u, v, k]["geometry"].coords)
                if cluster_label == u:
                    new_coords = xy + old_coords
                    deltalength = tbd.getdistance(
                        x, y, old_coords[0][0], old_coords[0][1]
                    )
                else:
                    new_coords = old_coords + xy
                    deltalength = tbd.getdistance(
                        old_coords[-1][0], old_coords[-1][1], x, y
                    )
                new_geom = LineString(new_coords)
                H.edges[u, v, k]["geometry"] = new_geom
                # update the edge length attribute, given the new geometry
                H.edges[u, v, k]["length"] = H.edges[u, v, k]["length"] + deltalength

        else:
            pass
    e_group = utils_graph.graph_to_gdfs(H, nodes=False).groupby(["u", "v"])
    F = nx.MultiDiGraph()
    F.graph = H.graph
    F.nodes = H.nodes

    for e_group, e_subset in e_group:
        edges_data = dict()
        if len(e_subset) == 1:
            edges_data = e_subset.to_dict(orient="records")[0]
            F.add_edge(e_group[0], e_group[1], **edges_data)
        else:
            edges_data["osmid"] = str(e_subset["osmid"].to_list())
            edges_data["oneway"] = str(e_subset["oneway"].to_list())
            edges_data["lanes"] = str(e_subset["lanes"].to_list())
            edges_data["name"] = str(e_subset["name"].to_list())
            edges_data["highway"] = str(e_subset["highway"].to_list())
            edges_data["from"] = e_group[0]
            edges_data["to"] = e_group[1]
            edges_data["length"] = e_subset["length"].max()
            edges_data["length_all"] = e_subset["length"].to_list()
            edges_data["from_original"] = str(e_subset["from_original"].to_list())
            edges_data["to_original"] = str(e_subset["to_original"].to_list())
            edges_data["maxspeed"] = e_subset["maxspeed"].astype(float).max()
            edges_data["access"] = str(e_subset["access"].to_list())
            edges_data["bridge"] = str(e_subset["bridge"].to_list())
            edges_data["tunnel"] = str(e_subset["tunnel"].to_list())
            edges_data["service"] = str(e_subset["service"].to_list())
            edges_data["geometry"] = e_subset["geometry"].unary_union
            F.add_edge(e_group[0], e_group[1], **edges_data)

    logs.info(f"new_edge form clusternode add to graph complete")

    null_nodes = [n for n, sc in F.nodes(data="street_count") if sc is None]
    street_count = stats.count_streets_per_node(F, nodes=null_nodes)
    nx.set_node_attributes(F, street_count, name="street_count")

    logs.info(f"clusterRoadnet complete")

    return F