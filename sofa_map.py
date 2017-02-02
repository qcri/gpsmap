"""
author: sofiane
Create the road network by merging trajectories.

"""
import geopy


class Cluster:
	def __init__(self, cid=None, nb_points=None, last_seen=None, lat=None, lon=None, angle=None):
		self.cid = cid
		self.lon = lon
		self.lat = lat
		self.angle = angle
		self.last_seen = last_seen
		self.nb_points = nb_points
		self.points = []

	def get_coordinates(self):
		return (self.lon, self.lat)

	def add(self, point):
		self.points.append(point)
		self.nb_points += 1
		self._recompute_center()

	def _recompute_center(self):
		self.lon = sum([p.lon for p in self.points]) / len(self.points)
		self.lat = sum([p.lat for p in self.points]) / len(self.points)
		self.angle = sum([p.angle for p in self.points]) / len(self.points)


from common import *
import math

RADIUS_METER = 25
RADIUS_DEGREE = RADIUS_METER * 10e-6
HEADING_ANGLE_TOLERANCE = 10
MAX_PATH_LEN = 5

INPUT_FILE_NAME = 'data/gps_data/gps_points_07-11.csv'
FILE_CODE = 'gps_points_07-11'
trajectories = create_trajectories(INPUT_FILE_NAME='data/gps_data/%s.csv' % FILE_CODE, waiting_threshold=21)

geodist = geopy.distance.VincentyDistance(meters=RADIUS_METER)


clusters = []
cluster_kdtree = None

roadnet = nx.DiGraph()

def satisfy_path_condition(G, s, t, max_path_len):
	"""
	I need to check if there's a path of length max_path_len or less. If so, return false.
	If true, then create an edge.
	To create an edge there shouldn't exist a path of length < max_path_len
	:param G:
	:param s:
	:param t:
	:param max_path_len:
	:return:
	"""
	path_lens = [len(path) for path in nx.all_simple_paths(G, s, t) if len(path) < max_path_len and len(path) > 1]
	print '\n len path: %s' % len(list(nx.all_simple_paths(G, s, t)))
	if len(path_lens) == 0:
		return True
	return False

for i, trajectory in enumerate(trajectories[:200]):
	sys.stdout.write('\rprocessing trajectory: %s' % i)
	sys.stdout.flush()

	prev_cluster = -1
	current_cluster = -1
	for point in trajectory:
		if len(clusters) == 0:
			# create a new cluster
			new_cluster = Cluster(cid=len(clusters), nb_points=1, last_seen=point.timestamp, lat=point.lat,
			                      lon=point.lon, angle=point.angle)
			clusters.append(new_cluster)
			roadnet.add_node(new_cluster.cid)
			current_cluster = new_cluster.cid  # all I need is the index of the new cluster
			# recompute the cluster index
			cluster_kdtree = cKDTree([c.get_coordinates() for c in clusters])
			continue

		# if there's a cluster within x meters and y angle: add to. Else: create new cluster
		close_clusters_indices = [clu_index for clu_index in cluster_kdtree.query_ball_point(x=point.get_coordinates(), r=RADIUS_DEGREE, p=2) if math.fabs(point.angle - clusters[clu_index].angle) <= HEADING_ANGLE_TOLERANCE ]
		if len(close_clusters_indices) == 0:
			# create a new cluster
			new_cluster = Cluster(cid=len(clusters), nb_points=1, last_seen=point.timestamp, lat=point.lat, lon=point.lon, angle=point.angle)
			clusters.append(new_cluster)
			roadnet.add_node(new_cluster.cid)
			current_cluster = new_cluster.cid
			# recompute the cluster index
			cluster_kdtree = cKDTree([c.get_coordinates() for c in clusters])

		else:
			# add the point to the cluster
			pt = geopy.Point(point.get_coordinates())
			close_clusters_distances = [geopy.distance.distance(pt, geopy.Point(clusters[clu_index].get_coordinates())).meters for clu_index in close_clusters_indices]
			closest_cluster_indx = close_clusters_indices[close_clusters_indices.index(min(close_clusters_indices))]
			clusters[closest_cluster_indx].add(point)
			current_cluster = closest_cluster_indx

		# create an edge between two consecutive clusters.
		if current_cluster != -1 and prev_cluster != -1 and prev_cluster != current_cluster and\
				(not current_cluster in roadnet.neighbors(prev_cluster)) and (satisfy_path_condition(roadnet, prev_cluster, current_cluster, MAX_PATH_LEN)):
				roadnet.add_edge(prev_cluster, current_cluster)
				prev_cluster = current_cluster
		elif prev_cluster == -1 and current_cluster != -1:
			prev_cluster = current_cluster

X = []
Y = []
for cluster in clusters:
	X.append(cluster.lon)
	Y.append(cluster.lat)

print '\nhere are %s clusters' % len(clusters)
print 'there are %s nodes' % roadnet.number_of_nodes()
print 'there are %s edges' % roadnet.number_of_edges()
plt.scatter(X, Y)

for s, t in roadnet.edges():
	plt.plot([clusters[s].lon, clusters[t].lon], [clusters[s].lat, clusters[t].lat], color='blue')

plt.show()




