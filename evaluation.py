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
from matplotlib import pyplot as plt
from common import build_road_network_from_shapefile_with_no_middle_nodes, remove_segments_with_no_points, \
	get_marble_holes, compute_approximate_precision_recall_f1, build_roadnet_from_edges_rss, build_roadnet_from_edges, \
	build_roadnet_from_edges_rss

import random
import random
import geopy.distance
from scipy.spatial import cKDTree


# 1. I have used QGIS to crop OSM map to the bbox of our data.

# 2. Create directed graph from OSM:
shapefile = 'data/shapefiles/relevant_osm_part_doha_roads.shp'
gt_rn = build_road_network_from_shapefile_with_no_middle_nodes(shape_file=shapefile)
print gt_rn.number_of_nodes(), gt_rn.number_of_edges()

# 3. TODO: implement the function that removes non supervised roads.
gt_rn = remove_segments_with_no_points(g=gt_rn, points=None)

# 4. build a directed graph for one of the algorithms:
edge_file = 'evaluation/cao_edges.txt'
test_rn = build_roadnet_from_edges(edge_fname=edge_file)

edge_file = 'evaluation/rss_edges.txt'
test_rn = build_roadnet_from_edges_rss(edge_fname=edge_file)

#for s, t in test_rn.edges()[:1000]:
#	plt.plot([s[0], t[0]], [s[1], t[1]], color='black')




# 5. marble/holes algorithm:
# 5.1. holes: generate a random position ==> lets pick a random node in the network.
nodes = gt_rn.nodes()
starting_point = nodes[random.randint(0, len(nodes) - 1)]
holes = get_marble_holes(rnet=gt_rn, radius=1000, frequency=5, starting_point=starting_point)

# 5.2. marbles: find closest node in test_rn to starting point.
nodes = test_rn.nodes()
dist = [geopy.distance.distance(geopy.Point(starting_point), geopy.Point(n)).meters for n in nodes]
indx_closest = dist.index(min(dist))
closest_starting_point = nodes[indx_closest]
marbles = get_marble_holes(rnet=test_rn, radius=1000, frequency=5, starting_point=closest_starting_point)

# 6. precision, recall, f1.
distance_threshold = 20 # in meters
precision, recall, f1 = compute_approximate_precision_recall_f1(marbles=marbles, holes=holes, distance_threshold=distance_threshold)


print 'GT nodes:', len(gt_rn.nodes())
print 'GT starting point:', starting_point
print 'GT Holes:', len(holes), holes[:10]

print '===================='
print 'Test nodes:', len(test_rn.nodes())
print 'Test starting point:', closest_starting_point
print 'Marbles:', len(marbles), marbles[:10]
print 'distance initial hole and marble:', geopy.distance.distance(geopy.Point(starting_point), geopy.Point(closest_starting_point)).meters
print 'precision: %s\trecall: %s\t f1: %s' % (precision, recall, f1)


# Plotting things for visualization:
#for s, t in gt_rn.edges():
#	plt.plot([s[0], t[0]], [s[1], t[1]], color='blue')

msizes = [5 for _ in range(len(holes))]
msizes [0] = 20
for hole in holes:
	plt.scatter([h[0] for h in holes], [h[1] for h in holes], marker='o', s=msizes)

#plt.scatter([starting_point[0]], [starting_point[1]], marker='*', color='red', s=20)

plt.axis('equal')
plt.show()