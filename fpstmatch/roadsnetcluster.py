import networkx as nx
import pandas as pd
import geopandas as gpd
import shapely
from sklearn.cluster import KMeans
import transbigdata as tbd

from . import distance
from . import logs
from . import settings
from . import stats
from . import utils
from . import utils_graph


def roadsnetcluster(orgRoadGraph, grid_rec):
    G = orgRoadGraph.copy()

    node_points = utils_graph.graph_to_gdfs(G, edges=False)
    nodeswithgrids = gpd.sjoin(node_points, grid_rec, how="left", predicate="within")
    nodeswithgrids = nodeswithgrids.drop(columns="geometry").rename(columns={"index_right": "cluster"})
    nodeswithgrids["cluster"] = nodeswithgrids["cluster"].factorize()[0]

    H = _gridnodescluster(G, nodeswithgrids)
    F = _combineroads(H)
    return H, F


def _gridnodescluster(G, nodeswithgrid):
    _H = nx.MultiDiGraph()
    _H.graph = G.graph

    groups = nodeswithgrid.groupby("cluster")
    for cluster_label, nodes_subset in groups:
        osmids_list = list(map(lambda x: str(x), nodes_subset.index.to_list()))
        osmids = ','.join(osmids_list)
        if len(osmids) == 1:
            # if cluster is a single node, add that node to new graph
            nodeswithgrid.loc[nodes_subset['osmid'].values[0], 'cluster'] = str(cluster_label) + '_0'
            _H.add_node(str(cluster_label) + '_0', osmid_original=osmids, **G.nodes[osmids[0]])
        else:
            # if cluster is multiple merged nodes, create one new node to
            # represent them
            cost = []  # 初始化损失（距离）值
            coords2cluster = nodes_subset[['x', 'y']].values.tolist()
            for i in range(1, len(coords2cluster)):  # 尝试不同的K值
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
                kmeans.fit(coords2cluster)
                cost.append(kmeans.inertia_)  # inerita_是我们选择的方法，其作用相当于损失函数
            if len(cost) < 3:
                clusterlen = 1
            else:
                cost_pd = pd.DataFrame({'cost': cost})
                x = cost_pd.index.tolist()
                y = cost_pd['cost'].values.tolist()
                kneedle = KneeLocator(x, y, S=1.0, curve='convex', direction='decreasing', interp_method="interp1d")
                if kneedle.elbow is None:
                    clusterlen = len(coords2cluster)
                else:
                    clusterlen = kneedle.elbow
            kmeansmodel = KMeans(n_clusters=clusterlen, init='k-means++')
            nodes_subset['cluster_sub'] = kmeansmodel.fit_predict(coords2cluster)

            groups_ = nodes_subset.groupby("cluster_sub")
            for cluster_label_, nodes_subset_ in groups_:
                osmids_list_ = list(map(lambda x: str(x), nodes_subset_.index.to_list()))
                osmids_ = ','.join(osmids_list_)
                nodeswithgrid.loc[nodes_subset_['osmid'].values.tolist(), 'cluster'] = str(cluster_label) + '_' + str(
                    cluster_label_)
                _H.add_node(
                    str(cluster_label) + '_' + str(cluster_label_),
                    osmid_original=osmids_,
                    x=kmeansmodel.cluster_centers_[cluster_label_][0],
                    y=kmeansmodel.cluster_centers_[cluster_label_][1],
                )

            # calculate street_count attribute for all nodes lacking it
            null_nodes = [n for n, sc in _H.nodes(data="street_count") if sc is None]
            street_count = stats.count_streets_per_node(_H, nodes=null_nodes)
            nx.set_node_attributes(_H, street_count, name="street_count")

    gdf_edges = utils_graph.graph_to_gdfs(G, nodes=False)
    for u, v, k, data in G.edges(keys=True, data=True):
        u2 = nodeswithgrid.loc[u, "cluster"]
        v2 = nodeswithgrid.loc[v, "cluster"]
        # only create the edge if we're not connecting the cluster
        # to itself, but always add original self-loops
        if (u2 != v2) or (u == v):
            data["from_original"] = u
            data["to_original"] = v
            data["from"] = u2
            data["to"] = v2
            if "geometry" not in data:
                data["geometry"] = gdf_edges.loc[(u, v, k), "geometry"]
            _H.add_edge(u2, v2, **data)

    groups_ = nodeswithgrid.groupby("cluster")
    for cluster_label, nodes_subset in groups_:
        # but only if there were multiple nodes merged together,
        # otherwise it's the same old edge as in original graph
        if len(nodes_subset) > 1:
            # get coords of merged nodes point centroid to prepend or
            # append to the old edge geom's coords
            # for each edge incident on this new merged node, update its
            # geometry to extend to/from the new node's point coords
            in_edges = set(_H.in_edges(cluster_label, keys=True))
            out_edges = set(_H.out_edges(cluster_label, keys=True))
            for u, v, k in in_edges | out_edges:
                x = _H.nodes[cluster_label]["x"]
                y = _H.nodes[cluster_label]["y"]
                xy = [(x, y)]
                old_coords = list(_H.edges[u, v, k]["geometry"].coords)
                if cluster_label == u:
                    new_coords = xy + old_coords
                    deltalength = tbd.getdistance(x, y, old_coords[0][0], old_coords[0][1])
                else:
                    new_coords = old_coords + xy
                    deltalength = tbd.getdistance(old_coords[-1][0], old_coords[-1][1], x, y)
                new_geom = shapely.geometry.LineString(new_coords)
                _H.edges[u, v, k]["geometry"] = new_geom
                # update the edge length attribute, given the new geometry
                _H.edges[u, v, k]["length"] = _H.edges[u, v, k]['length'] + deltalength
        else:
            pass
    return _H


def _combineroads(H):
    _F = nx.MultiDiGraph()
    _F.graph = H.graph
    _F.nodes = H.nodes
    e_group = utils_graph.graph_to_gdfs(H, nodes=False).reset_index().groupby(['u', 'v'])
    for e_group, e_subset in e_group:
        edges_data = dict()
        if len(e_subset) == 1:
            # e_subset.reindex()
            e_subset.drop(columns=['u', 'v', 'key'], inplace=True)
            edges_data = e_subset.to_dict(orient='records')[0]
            _F.add_edge(e_group[0], e_group[1], **edges_data)
        else:
            edges_data['osmid'] = str(e_subset['osmid'].to_list())
            edges_data['oneway'] = str(e_subset['oneway'].to_list())
            edges_data['lanes'] = str(e_subset['lanes'].to_list())
            edges_data['name'] = str(e_subset['name'].to_list())
            edges_data['highway'] = str(e_subset['highway'].to_list())
            edges_data['from'] = e_group[0]
            edges_data['to'] = e_group[1]
            edges_data['length'] = e_subset['length'].max()
            edges_data['from_original'] = str(e_subset['from_original'].to_list())
            edges_data['to_original'] = str(e_subset['to_original'].to_list())
            edges_data['maxspeed'] = e_subset['maxspeed'].astype(float).max()
            edges_data['access'] = str(e_subset['access'].to_list())
            edges_data['bridge'] = str(e_subset['bridge'].to_list())
            edges_data['tunnel'] = str(e_subset['tunnel'].to_list())
            edges_data['service'] = str(e_subset['service'].to_list())
            edges_data['geometry'] = e_subset['geometry'].unary_union
            _F.add_edge(e_group[0], e_group[1], **edges_data)
    return _F
