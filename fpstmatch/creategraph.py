"""Graph creation functions."""
import copy

import networkx as nx
import shapely

from . import distance, logs, settings, stats, utils, utils_graph


def graph_from_geopandas_(nodes_info, edges_info, ignore_flag=None):
    metadata = {
        "created_date": utils.ts(),
        "created_with": f"OSMnx {settings.__version__}",
        "crs": settings.default_crs,
    }
    _G = nx.MultiDiGraph(**metadata)
    for node_grid, node_info in nodes_info.iterrows():
        _G.add_node(node_grid, **node_info)
    for edge_grid, edge_info in edges_info.iterrows():
        if ignore_flag is not None:
            if not edge_info[ignore_flag]:
                _G.add_edges_from([list(edge_grid)], **edge_info)
        else:
            _G.add_edges_from([list(edge_grid)], **edge_info)
    return _G


def graph_from_detectors_(
    nodes_gdf,
    edges_gdf,
    bidirectional=False,
):
    metadata = {
        "created_date": utils.ts(),
        "created_with": f"OSMnx {settings.__version__}",
        "crs": settings.default_crs,
    }
    G = nx.MultiDiGraph(**metadata)

    for i, each_node in nodes_gdf.iterrows():
        node = each_node.to_dict()
        node["y"] = node["lat"]
        node["x"] = node["lng"]
        G.add_node(node["osmid"], **node)

    for i, each_edge in edges_gdf.iterrows():
        path = each_edge.to_dict()

        path["nodes"] = [path["from"], path["to"]]
        _add_path(G, [], path, bidirectional=bidirectional, extendflag=True)

    return G


def graph_from_(
    nodes_gdf,
    edges_gdf,
    network_type="all_private",
    simplify=True,
    retain_all=False,
    truncate_by_edge=False,
    clean_periphery=True,
    great_circle_distance=False,
):
    if clean_periphery:
        # create buffered graph from the downloaded data
        bidirectional = network_type in settings.bidirectional_network_types
        G = _create_graph(
            nodes_gdf,
            edges_gdf,
            retain_all=retain_all,
            bidirectional=bidirectional,
            great_circle_distance=great_circle_distance,
        )

        # count how many physical streets in buffered graph connect to each
        # intersection in un-buffered graph, to retain true counts for each
        # intersection, even if some of its neighbors are outside the polygon
        spn = stats.count_streets_per_node(G, nodes=G.nodes)
        nx.set_node_attributes(G, values=spn, name="street_count")
        logs.info(
            f"graph_from_polygon returned graph with {len(G)} nodes and {len(G.edges)} edges"
        )
        return G
    else:
        logs.info(f"clean_periphery is {clean_periphery}")
        return


def _create_graph(
    nodes_gdf,
    edges_gdf,
    retain_all=False,
    bidirectional=False,
    great_circle_distance=False,
):
    """
    Create a networkx MultiDiGraph from GeoPandas.

    Adds length attributes in meters (great-circle distance between endpoints)
    to all of the graph's (pre-simplified, straight-line) edges via the
    `distance.add_edge_lengths` function.

    Parameters
    ----------
    retain_all : bool
        if True, return the entire graph even if it is not connected.
        otherwise, retain only the largest weakly connected component.
    bidirectional : bool
        if True, create bi-directional edges for one-way streets

    Returns
    -------
    G : networkx.MultiDiGraph
    """
    logs.info("Creating graph from GeoPandas postgresql OSM data...")

    # create the graph as a MultiDiGraph and set its meta-attributes
    metadata = {
        "created_date": utils.ts(),
        "created_with": f"OSMnx {settings.__version__}",
        "crs": settings.default_crs,
    }
    G = nx.MultiDiGraph(**metadata)

    # extract nodes and paths from the downloaded osm data
    # nodes = dict()
    # paths = dict()
    nodes = list()
    paths = list()
    nodes_idset = list()

    for i, each_node in nodes_gdf.iterrows():
        nodes.append(_convert_node(each_node))

    for i, each_edge in edges_gdf.iterrows():
        paths.append(_convert_path(each_edge))

    # add each osm node to the graph
    for node in nodes:
        nodes_idset.append(node["osmid"])
        G.add_node(node["osmid"], **node)

    # add each osm way (ie, a path of edges) to the graph
    for path_data in paths:
        _add_path(G, nodes_idset, path_data, bidirectional)

    # retain only the largest connected component if retain_all is False
    if not retain_all:
        G = utils_graph.get_largest_component(G, strongly=True)

    # add length (great-circle distance between nodes) attribute to each edge
    if great_circle_distance:
        if len(G.edges) > 0:
            G = distance.add_edge_lengths(G)

    logs.info(f"Created graph with {len(G)} nodes and {len(G.edges)} edges")

    return G


def _convert_node(each_node):
    """
    Convert an OSM node element into the format for a networkx node.

    Parameters
    ----------
    Returns
    -------
    node : dict
    """

    node = {"y": each_node["lat"],
            "x": each_node["lng"], "osmid": each_node["osmid"]}
    for useful_tag in settings.useful_tags_node:
        if useful_tag in each_node.index:
            node[useful_tag] = each_node[useful_tag]
    return node


def _convert_path(each_edge, eachosmid=None):
    """
    Convert an OSM way element into the format for a networkx path.

    Parameters
    ----------

    Returns
    -------
    path : dict
    """

    path = {"osmid": each_edge["osmid"], "nodes": [
        each_edge["from"], each_edge["to"]]}
    if each_edge["geometry"].geom_type == "MultiLineString":
        path["geometry"] = shapely.ops.linemerge(each_edge["geometry"])
    elif each_edge["geometry"].geom_type == "LineString":
        path["geometry"] = each_edge["geometry"]
    for useful_tag in settings.useful_tags_way:
        if useful_tag in each_edge.index:
            path[useful_tag] = each_edge[useful_tag]

    return path


def _parse_nodes(each_node):
    """
    Construct dicts of nodes and paths from an Overpass response.

    Parameters
    ----------

    Returns
    -------
    nodes, paths : tuple of dicts
        dicts' keys = osmid and values = dict of attributes
    """
    # nodes = dict()
    # nodes[each_node["osmid"]] = _convert_node(each_node)
    nodes = _convert_node(each_node)
    return nodes


def _parse_paths(each_edge, eachosmid=None):
    """
    Construct dicts of nodes and paths from an Overpass response.

    Parameters
    ----------

    Returns
    -------
    nodes, paths : tuple of dicts
        dicts' keys = osmid and values = dict of attributes
    """
    # paths = dict()
    # paths[each_edge["fid"]] = _convert_path(each_edge, eachosmid)
    paths = _convert_path(each_edge, eachosmid)
    return paths


def _is_path_one_way(path, bidirectional, oneway_values):
    """
    Determine if a path of nodes allows travel in only one direction.

    Parameters
    ----------
    path : dict
        a path's tag:value attribute data
    bidirectional : bool
        whether this is a bi-directional network type
    oneway_values : set
        the values OSM uses in its 'oneway' tag to denote True

    Returns
    -------
    bool
    """
    # rule 1
    if settings.all_oneway:
        # if globally configured to set every edge one-way, then it's one-way
        # print("rule 1")
        return True

    # rule 2
    elif bidirectional:
        # if this is a bi-directional network type, then nothing in it is
        # considered one-way. eg, if this is a walking network, this may very
        # well be a one-way street (as cars/bikes go), but in a walking-only
        # network it is a bi-directional edge (you can walk both directions on
        # a one-way street). so we will add this path (in both directions) to
        # the graph and set its oneway attribute to False.
        # print("rule 2")
        return False

    # rule 3
    elif "oneway" in path and path["oneway"] in oneway_values:
        # if this path is tagged as one-way and if it is not a bi-directional
        # network type then we'll add the path in one direction only
        # print("rule 3")
        return True

    # rule 4
    elif "junction" in path and path["junction"] == "roundabout":
        # roundabouts are also one-way but are not explicitly tagged as such
        # print("rule 4")
        return True

    else:
        # otherwise this path is not tagged as a one-way
        # print("rule 5")
        return False


def _is_path_reversed(path, reversed_values):
    """
    Determine if the order of nodes in a path should be reversed.

    Parameters
    ----------
    path : dict
        a path's tag:value attribute data
    reversed_values : set
        the values OSM uses in its 'oneway' tag to denote travel can only
        occur in the opposite direction of the node order

    Returns
    -------
    bool

    """

    if "oneway" in path and path["oneway"] in reversed_values:
        return True
    else:
        return False


def _add_path(G, nodes_idset, path, bidirectional=False, extendflag=False):
    nodes_ = path.pop("nodes")

    if not extendflag:
        if nodes_[0] not in nodes_idset or nodes_[1] not in nodes_idset:
            return

    # reverse the order of nodes in the path if this path is both one-way
    # and only allows travel in the opposite direction of nodes' order
    is_one_way = _is_path_one_way(path, bidirectional, settings.oneway_values)

    G.add_edges_from([nodes_], **path)
    if not is_one_way and _is_path_reversed(path, settings.oneway_values):
        nodes_.reverse()
        temp = list(path["geometry"].coords)
        temp.reverse()
        path["geometry"] = shapely.geometry.LineString(temp)
        G.add_edges_from([nodes_], **path)
