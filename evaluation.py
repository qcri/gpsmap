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
# edge_file = 'evaluation/cao_edges.txt'
# test_rn = build_roadnet_from_edges(edge_fname=edge_file)

edge_file = 'evaluation/rss_edges_v1.txt'
test_rn = build_roadnet_from_edges_rss(edge_fname=edge_file)

# 5. marble/holes algorithm:
# 5.1. holes: generate a random position ==> lets pick a random node in the network.
gt_nodes = gt_rn.nodes()
nb_runs = 20
distance_thresholds = [5, 10, 15, 20, 25] # in meters
holes_marbles_interval = 5 # meters
holes_marbles_radius = 2000 # meters

global_missings = []
global_spuriouses = []
global_f1s = []
for distance_threshold in distance_thresholds:
	missings = []
	spuriouses = []
	f1s = []
	for i in range(nb_runs):
		starting_point = gt_nodes[random.randint(0, len(gt_nodes) - 1)]
		holes = get_marble_holes(rnet=gt_rn, radius=1000, frequency=holes_marbles_interval, starting_point=starting_point)

		# 5.2. marbles: find closest node in test_rn to starting point.
		nodes = test_rn.nodes()
		dist = [geopy.distance.distance(geopy.Point(starting_point), geopy.Point(n)).meters for n in nodes]
		indx_closest = dist.index(min(dist))
		closest_starting_point = nodes[indx_closest]
		marbles = get_marble_holes(rnet=test_rn, radius=1000, frequency=holes_marbles_interval, starting_point=closest_starting_point)
		# print 'nb_holes:%s\t nb_marbles:%s' % (len(holes), len(marbles))
		# 6. precision, recall, f1.
		spurious, missing, f1 = compute_approximate_precision_recall_f1(marbles=marbles, holes=holes, distance_threshold=distance_threshold)
		print 'distance:%s \tspurious: %s\tmissing: %s\t f1: %s' % (distance_threshold, spurious, missing, f1)
		spuriouses.append(spurious)
		missings.append(missing)
		f1s.append(f1)
	global_spuriouses.append(sum(spuriouses) / len(spuriouses))
	global_missings.append(sum(missings) / len(missings))
	global_f1s.append(sum(f1s) / len(f1s))

with open('data/f1_scores_s3r_v1.txt', 'w') as f:
	for dis, sp, mi, f1 in zip(distance_thresholds, global_spuriouses, global_missings, global_f1s):
		f.write('%s,%s,%s,%s\n' % (dis, sp, mi, f1))
#
# distance_thresholds = []
# global_f1s = []
# with open('data/f1_scores_s3r_v1.txt', 'r') as f:
# 	for line in f:
# 		distance_thresholds.append(float(line.split(',')[0]))
# 		global_f1s.append(float(line.split(',')[3]))
plt.plot(distance_thresholds, global_f1s, marker='o', color='black')
plt.ylabel('F-score')
plt.xlabel('Matching distance threshold (m)')
plt.legend(['R3S'])
plt.savefig('figs/f1_scores_s3r_v1.png', format='PNG')


print 'GT nodes:', len(gt_rn.nodes())
print 'GT starting point:', starting_point
print 'GT Holes:', len(holes), holes[:10]

print '===================='
print 'Test nodes:', len(test_rn.nodes())
print 'Test starting point:', closest_starting_point
print 'Marbles:', len(marbles), marbles[:10]
print 'distance initial hole and marble:', geopy.distance.distance(geopy.Point(starting_point), geopy.Point(closest_starting_point)).meters
print 'spurious: %s\tmissing: %s\t f1: %s' % (spurious, missing, f1)
# print 'precision: %s\trecall: %s\t f1: %s' % (spurious, missing, f1)


# =====================================================================
# Plotting things for visualization:
# =====================================================================
#
# #for s, t in gt_rn.edges():
# #	plt.plot([s[0], t[0]], [s[1], t[1]], color='black')
#
# #for s, t in test_rn.edges():
# #	plt.plot([s[0], t[0]], [s[1], t[1]], color='blue')
#
# hsizes = [5 for _ in range(len(holes))]
# hsizes[0] = 100
# plt.scatter([h[0] for h in holes], [h[1] for h in holes], marker='o', color='blue', s=hsizes)
#
# msizes = [5 for _ in range(len(marbles) + 1)]
# msizes[0] = 100
# marbles = [closest_starting_point] + marbles
# plt.scatter([h[0] for h in marbles], [h[1] for h in marbles], marker='o', color='red', s=msizes)
# plt.axis('equal')
# plt.show()