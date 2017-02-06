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

RADIUS_METER = 20
RADIUS_DEGREE = RADIUS_METER * 10e-6
HEADING_ANGLE_TOLERANCE = 10
MAX_PATH_LEN = 5
MAX_PATH_DISTANCE_FACTOR = 1.3

INPUT_FILE_NAME = 'data/gps_data/gps_points_07-11.csv'
FILE_CODE = 'gps_points_07-11'
trajectories = create_trajectories(INPUT_FILE_NAME='data/gps_data/%s.csv' % FILE_CODE, waiting_threshold=21)

geodist = geopy.distance.VincentyDistance(meters=RADIUS_METER)


clusters = []
cluster_kdtree = None

roadnet = nx.DiGraph()


def k_reachable_from(g, k):
	"""
	Build a dictionary that returns for each node, all the nodes from which it can be reached within radius k.
	:param g:
	:param k:
	:return:
	"""
	nodes = g.nodes()
	k_reach_from = defaultdict(set)
	for node in nodes:
		sub_g = nx.ego_graph(g, node, radius=k)
		for n in sub_g.nodes():
			k_reach_from[n].add(node)
	return k_reach_from


def update_k_reachability(source, roadnet, k, k_reach):
	"""
	In order to update the k_reachability of only those nodes that will be impacted by the creation of a new
	edge: s --> t.
	:param s: source node of the new edge
	:param t: target node of the new edge
	:param g: the road graph
	:param k: reachability radius
	:param k_reach: dictionary of k_reachability to be updated
	:return: updated k_reachability
	"""
	inv_g = roadnet.reverse()
	inv_reachability = nx.ego_graph(inv_g, source, radius=k)
	for node in inv_reachability:
		k_reach[node] = nx.ego_graph(roadnet, node, radius=k).nodes()
	return k_reach

def k_reachability(g, k):
	"""
	Build a dictionary in which we return for each node, all nodes that we can reach within k hops.
	:param g: graph
	:param k: number of hops
	:return: dictionary
	"""
	nodes = g.nodes()
	k_reach = defaultdict(list)
	for node in nodes:
		sub_g = nx.ego_graph(g, node, radius=k)
		k_reach[node] = sub_g.nodes()
	return k_reach



def satisfy_path_condition(s, t, k_reach):
	"""
	I need to check if there's a path of length max_path_len or less. If so, return false.
	If true, then create an edge.
	To create an edge there shouldn't exist a path of length < max_path_len
	:param s:
	:param t:
	:param k_reach:
	:return:
	"""
	if s == -1 or t == -1 or s == t or (s in k_reach.keys() and t in k_reach[s]):
		return False
	return True

def satisfy_path_condition_distance(s, t, g, clusters, max_length_distance_factor):
	"""
	return False if there's a path of length max length, True otherwise
	:param s:
	:param t:
	:param k_reach:
	:return:
	"""

	if s == -1 or t == -1:
		return False
	edge_distance = geopy.distance.distance(geopy.Point(clusters[s].get_coordinates()),\
	                                        geopy.Point(clusters[t].get_coordinates())).meters

	print 'source, target;', s, t
	path = nx.shortest_path(g, source=s, target=t)
	path_length_meters = 0
	for i in range(1, len(path)):
		path_length_meters += geopy.distance.distance(geopy.Point(clusters[path[i - 1]].get_coordinates()),\
	                                        geopy.Point(clusters[path[i]].get_coordinates())).meters

	if path_length_meters <= max_length_distance_factor * edge_distance:
		return False
	return True

for i, trajectory in enumerate(trajectories[:200]):
	sys.stdout.write('\rprocessing trajectory: %s' % i)
	sys.stdout.flush()

	prev_cluster = -1
	current_cluster = -1
	k_path_length = 5
	k_reach = defaultdict(list)
	first_edge = True
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

		if satisfy_path_condition_distance(prev_cluster, current_cluster, roadnet, clusters, MAX_PATH_DISTANCE_FACTOR):
				roadnet.add_edge(prev_cluster, current_cluster)
				prev_cluster = current_cluster
				# if first_edge:
				# 	k_reach = k_reachability(roadnet, k_path_length)
				# 	first_edge = False
				# # else:
				# 	# k_reach = update_k_reachability(prev_cluster, roadnet, k_path_length, k_reach)
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




