
import networkx as nx
from matplotlib import pyplot as plt
from common import remove_segments_with_no_points, build_road_network_from_shapefile_with_no_middle_nodes


file_points = 'data/gps_data/gps_points_07-11.csv'
data = []
X = []
Y = []
with open(file_points, 'r') as f:
	for line in f:
		x = line.split('\t')
		data.append((float(x[0]), float(x[1])))
		X.append(float(x[0]))
		Y.append(float(x[1]))
# 2. Create directed graph from OSM:
shapefile = 'data/shapefiles/relevant_osm_part_doha_roads.shp'
gt_rn = build_road_network_from_shapefile_with_no_middle_nodes(shape_file=shapefile)
nseg = gt_rn.number_of_edges()
nx.write_gpickle(gt_rn, 'data/doha_roads.gpickle')

for s, t in gt_rn.edges():
	plt.plot([s[0], t[0]], [s[1], t[1]], color='blue')

clean_rn = remove_segments_with_no_points(rn=gt_rn, data=data, distance_threshold=50)
nx.write_gpickle(gt_rn, 'data/clean_doha_roads.gpickle')

print 'segments in initial network:%s, segments in new network:%s' % (nseg, clean_rn.number_of_edges())


plt.scatter(X,Y, color='black')
for s, t in clean_rn.edges():
	plt.plot([s[0], t[0]], [s[1], t[1]], color='red')

plt.show()