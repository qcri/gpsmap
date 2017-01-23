# author: sofiane
# date: 2017-01-17
#
# This code is for evaluating maps.
# For now we use OSM maps as ground truth, we'll change it later to QMIC map
# steps:
# 1. get OSM with the right corresponding bbox
# 3. build a directed graph from OSM
# 3. remove roads for wich we don't have data
# 4. build a directed graph from other algorithms' maps
# 5. implement marbles/holes evaluation algorithm
# 6. report precision, recall, f1.
#

import networkx as nx
from common import build_road_network_from_shapefile_with_no_middle_nodes, remove_segments_with_no_points
from interface_cao2009 import build_roadnet_from_edges

# 1. I have used QGIS to crop OSM map to the bbox of our data.

# 2. Create directed graph from OSM:
shapefile = 'data/shapefiles/relevant_osm_part_doha_roads.shp'
gt_rn = build_road_network_from_shapefile_with_no_middle_nodes(shape_file=shapefile)
print gt_rn.number_of_nodes(), gt_rn.number_of_edges()

# 3. TODO: implement the function that removes non supervised roads.
gt_rn = remove_segments_with_no_points(g=gt_rn, points=None)

# 4. build a directed graph for one of the algorithms:
edge_file = 'map_inference_algorithms/cao_edges.txt'
test_rn = build_roadnet_from_edges(edge_fname=edge_file)

# 5. marble/holes algorithm:
holes = get_marble_holes(rnet=gt_rn, radius=1000, frequency=5)

print gt_rn.nodes()
print '===================='
print test_rn.nodes()


