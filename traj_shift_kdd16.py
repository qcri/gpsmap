import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from math import tan, sin, cos, asin, radians, exp, sqrt, ceil, atan, atan2, pow, degrees
from scipy.spatial import cKDTree
import datetime
import random
import json
import operator
import networkx as nx

BIN_SEGMENT_LENGTH = 2  # length of bins in meters


class GpsPoint:
	def __init__(self, data=None):
		if data is not None:
			self.btid = data[4]
			self.speed = float(data[5])
			self.timestamp = datetime.datetime.strptime(data[6], '%Y-%m-%d %H:%M:%S+03')
			self.lon = float(data[7])
			self.lat = float(data[8])
			self.angle = float(data[9])

	def get_coordinates(self):
		"""
		return the lon,lat of a gps point
		:return: tuple (lon, lat)
		"""
		return (self.lon, self.lat)

	def __str__(self):
		return "bt_id:%s, speed:%s, timestamp:%s, lon:%s, lat:%s, angle:%s" % \
		       (self.btid, self.speed, self.timestamp, self.lon, self.lat, self.angle)

	def __repr__(self):
		return "bt_id:%s, speed:%s, timestamp:%s, lon:%s, lat:%s, angle:%s" % \
		       (self.btid, self.speed, self.timestamp, self.lon, self.lat, self.angle)


class SamplePoint:
	def __init__(self, speed=None, lon=None, lat=None, angle=None, weight=None):
		self.speed = speed
		self.lon = lon
		self.lat = lat
		self.angle = angle
		self.weight = weight

	def get_coordinates(self):
		"""
		return the lon,lat of a gps point
		:return: tuple (lon, lat)
		"""
		return (self.lon, self.lat)

	def __str__(self):
		return "weight:%s, speed:%s, lon:%s, lat:%s, angle:%s" % \
		       (self.weight, self.speed, self.lon, self.lat, self.angle)

	def __repr__(self):
		return "weight:%s, speed:%s, lon:%s, lat:%s, angle:%s" % \
		       (self.weight, self.speed, self.lon, self.lat, self.angle)


class line:
	def __init__(self, slope=None, intercept=None):
		self.slope = slope if slope is not None else None
		self.intercept = intercept if intercept is not None else None

	def perpendecular(self, at_pt=None):
		if self.slope == 0:
			# if slope is 0 (horizontal line), the perpendicular is special: x=pt[0].
			# Thus, I'm givin a special interpretation to slope and intercept here to capture this case.
			# TODO: find a better solution to handle this case.
			return line(slope=float('Inf'), intercept=at_pt[0])
		slope = -1 / self.slope
		intercept = at_pt[1] - slope * at_pt[0] if at_pt is not None else 0
		return line(slope=slope, intercept=intercept)

	def plot(self, color=None, width=2, pt=None):
		x = np.array([-0.00002, -0.0001, 0, 0.00001, 0.00002]) + pt[0]
		if color is not None:
			plt.plot(x, self.intercept + self.slope * x, linewidth=width, color=color)
		else:
			plt.plot(x, self.intercept + self.slope * x, linewidth=width)

	def __repr__(self):
		return 'y = %sx + %s' % (self.slope, self.intercept)


def project_point_into_line(pt, parallel_line, perpendicular_line):
	"""
    This functions finds the projected point (ppt) of a point (pt) on a line (line). It computed the intersection
    between line at the perpondecular line to it that goes through pt. 
    :param pt: point (x, y)
    :param line: line defined with (intercept, slope)
    :returns: intersection point (x, y)
    """

	if perpendicular_line.slope == float('Inf'):
		# special case: projecting on a vertical line, y=point's y, x=intercept that has special meaning here.
		return (perpendicular_line.intercept, pt[1])
	# 1. find the line that is parallel to parallel line that goes thu pt.
	intercept = pt[1] - parallel_line.slope * pt[0]
	# line(parallel_line.slope, intercept).plot(color='green', width=1)

	# 2. find intersection point between pt_line and perpendicular_line
	X = np.array([[1, - parallel_line.slope], [1, -perpendicular_line.slope]])
	Y = np.array([intercept, perpendicular_line.intercept])
	yx = np.linalg.solve(X, Y)
	return (yx[1], yx[0])


def line_of_gps_point(pt, angle):
	"""
    Generate the line of a gps point
    :return: line object
    
    """
	b = tan(radians(90 - angle))
	a = pt[1] - b * pt[0]
	return line(intercept=a, slope=b)


def distance_heading_speed(pt1, pt2, sigma_h=0.5, sigma_s=0.5):
	delta_s = exp(-(pt1.speed - pt2.speed) ** 2 / sigma_s)
	delta_h = exp(-(pt1.angle - pt2.angle) ** 2 / sigma_h)
	return delta_s * delta_h


def haversine(pt1, pt2):
	"""
	Calculate the great circle distance between two points
	on the earth (specified in decimal degrees)
	Sofiane: got it from:http://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
	:param pt1: point (lon, lat)
	:param pt2: point (lon, lat)
	:return: the distance in meters
	"""

	lon1 = pt1[0]
	lat1 = pt1[1]
	lon2 = pt2[0]
	lat2 = pt2[1]

	# convert decimal degrees to radians
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

	# haversine formula
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
	c = 2 * asin(sqrt(a))
	km = 6367 * c
	return km * 1000


def load_data(fname='data/gps_data/gps_points.csv'):
	"""
	Given a file that contains gps points, this method creates different data structures
	:param fname: the name of the input file, as generated by QMIC
	:return: data_points (list of gps positions with their metadata), raw_points (coordinates only),
	points_tree is the KDTree structure to enable searching the points space
	"""
	data_points = list()
	raw_points = list()

	with open(fname, 'r') as f:
		f.readline()
		for line in f:
			tup = line.strip().split('\t')
			pt = GpsPoint(tup)
			data_points.append(pt)
			raw_points.append(pt.get_coordinates())
	points_tree = cKDTree(raw_points)
	return np.array(data_points), np.array(raw_points), points_tree


def _to_geojson(samples):
	"""
	Generate the geojson object of a list of sample points
	:param samples: list of samples
	:return: one geojson object that contains all the points.
	"""
	geojson = {'type': 'FeatureCollection', 'features': []}
	for s in samples:
		feature = {'type': 'Feature', 'properties': {}, 'geometry': {'type': 'Point', 'coordinates': []}}
		feature['geometry']['coordinates'] = [s.lon, s.lat]
		feature['properties']['speed'] = s.speed
		feature['properties']['weight'] = s.weight
		feature['properties']['angle'] = s.angle
		geojson['features'].append(feature)
	return geojson


def segments_to_geojson(segments, g):
	"""
	Generate the geojson object of a list of segments
	:param samples: list of samples
	:return: one geojson object that contains all the points.
	"""
	geojson = {'type': 'FeatureCollection', 'features': []}
	for s in segments:
		feature = {'type': 'Feature', 'properties': {}, 'geometry': {'type': 'LineString', 'coordinates': []}}
		coordinates = [[g.node[p]['lon'], g.node[p]['lat']] for p in list(s)]
		feature['geometry']['coordinates'] = coordinates
		geojson['features'].append(feature)
	return geojson


def _to_samplepoints(geojson):
	samples = []
	for s in geojson['features']:
		if s['geometry']['coordinates'][0] == 0 or s['geometry']['coordinates'][1] == 0: continue
		samples.append(SamplePoint(speed=s['properties']['speed'], angle=s['properties']['angle'],
		                           weight=s['properties']['weight'], lon=s['geometry']['coordinates'][0],
		                           lat=s['geometry']['coordinates'][1]))
	return samples


def retrieve_neighbors(in_pt, points_tree, radius=0.005):
	"""
	Retrieve all points within a radius to an input point in_pt
	:param in_pt: intput point
	:param points_tree: KDTree of points, needs to be precomputed
	:param radius: distance radius
	:return: list of indexes of neighbors.
	"""

	neighbors = points_tree.query_ball_point(x=in_pt, r=radius, p=2)
	return neighbors


def find_sample(rand_pt=None, neighbors=None, draw_output=False):
	if len(neighbors) < 2:
		return SamplePoint(lon=rand_pt.lon, lat=rand_pt.lat, speed=rand_pt.speed, angle=rand_pt.angle,
		                   weight=len(neighbors))

	in_pt = rand_pt.get_coordinates()
	angle = rand_pt.angle

	# equation of the heading line of the point:
	eq_heading = line_of_gps_point(in_pt, angle)
	# equation of the perpendicular line to the point's heading line:
	eq_perpend = eq_heading.perpendecular(at_pt=in_pt)
	# print 'heading eq:', eq_heading
	# print 'perpendecular eq:', eq_perpend

	# Project all points into the perpendicular line
	projected_points = list()
	for pt in neighbors:
		intersec_pt = project_point_into_line(pt.get_coordinates(), eq_heading, eq_perpend)
		projected_points.append(intersec_pt)

	# Create bins: my trick is the following:
	# consider x-s and y-s of the projected points.
	# compute different y_max-y_min and x_max-x_min, consider which ever axis maximazed this difference.
	# work on that axis only and split it into chunks of equal distances? or same number of chunks!

	xs, ys = [_[0] for _ in projected_points], [_[1] for _ in projected_points]
	minx, maxx = min(xs), max(xs)
	miny, maxy = min(ys), max(ys)
	axis_label = ''
	if maxx - minx >= maxy - miny:
		axis = xs
		axis_label = 'x-axis'
	else:
		axis = ys
		axis_label = 'y-axis'

	# figure out number of bins: this is based on the width of road lanes!
	LANE_WIDTH = 3.3  # lane width in meters
	segment_length = max([haversine(neighbors[i].get_coordinates(), neighbors[j].get_coordinates()) \
	                      for i in range(len(neighbors)) for j in range(i + 1, len(neighbors))])
	nb_bins = int(ceil(segment_length / LANE_WIDTH))
	# Handle the case where nb_bins = 0 / this happens if all points are superposed (collision)
	if nb_bins == 0:
		nb_bins = 1
	histog = np.histogram(axis, bins=nb_bins, density=True)
	densities, bin_limits = histog[0], histog[1]

	# Find the relevant/central bin:
	# common_denominator = sum([distance_heading_speed(rand_pt, nei) for nei in neighbors])
	bin_votes = []
	for i, (bmin, bmax) in enumerate(zip(bin_limits[:-1], bin_limits[1:])):
		# find neighbors that are inside the bin limits
		votes = 0
		for nei in neighbors:
			if (axis_label == 'x-axis' and nei.lon >= bmin and nei.lon < bmax) or \
					(axis_label == 'y-axis' and nei.lat >= bmin and nei.lat < bmax):
				votes += distance_heading_speed(rand_pt, nei)
		bin_votes.append(votes)
	# bin_votes.append(votes/common_denominator)
	max_density_bin = np.argmax(bin_votes)

	# max_density_bin = np.argmax(densities)
	marker = (bin_limits[max_density_bin] + bin_limits[max_density_bin + 1]) / 2
	# find the other component of the sample point: parallel to x-axis or y-axis
	if axis_label == 'x-axis':
		Sx = marker
		Sy = eq_perpend.intercept + eq_perpend.slope * Sx
	else:
		Sy = marker
		Sx = (Sy - eq_perpend.intercept) / eq_perpend.slope

	if draw_output == False:
		avg_speed = sum([nei.speed for nei in neighbors]) / len(neighbors)
		avg_heading = sum([nei.angle for nei in neighbors]) / len(neighbors)
		return SamplePoint(lon=Sx, lat=Sy, speed=avg_speed, angle=avg_heading, weight=len(neighbors))

	# ---------------------
	# plotting the results |
	# ---------------------

	eq_heading.plot(color='blue', pt=in_pt)
	eq_perpend.plot(color='red', pt=in_pt)
	plt.scatter(in_pt[0], in_pt[1], marker='8', s=50, color='r')

	for pt, intersec_pt in zip(neighbors, projected_points):
		plt.scatter(intersec_pt[0], intersec_pt[1], marker='*', s=10)
		plt.scatter(pt.lon, pt.lat, marker='o')
	plt.scatter(Sx, Sy, marker='s', s=50, color='green')

	xs, ys = [_.lon for _ in neighbors], [_.lat for _ in neighbors]
	minx, maxx = min(xs), max(xs)
	miny, maxy = min(ys), max(ys)
	plt.xlim([minx - 0.00001, maxx + 0.00001])
	plt.ylim([miny - 0.00001, maxy + 0.00001])
	# Get x/y-axis in the same aspect
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()
	return SamplePoint(lon=Sx, lat=Sy, speed=rand_pt.speed, angle=rand_pt.angle, weight=len(neighbors))


def traj_meanshift_sampling():
	INPUT_FILE_NAME = 'data/gps_data/gps_points_07-11.csv'
	RADIUS = 0.0002
	ANGLE_TOLERANCE = 60  # the tolerance of angle to consider two points are being heading toward the same overall direction
	data_points, raw_points, points_tree = load_data(fname=INPUT_FILE_NAME)
	Npoints = len(data_points)
	print 'nb points:', len(data_points), 'points example:', data_points[0], raw_points[0]

	samples = list()
	removed_points = set()
	available_point_indexes = np.arange(0, len(data_points))
	while len(available_point_indexes) > 0:
		print 'THERE ARE %s points, removed %s points' % (len(available_point_indexes), len(removed_points))
		rand_index = random.sample(available_point_indexes, 1)[0]
		rand_pt = data_points[rand_index]

		# Find all neighbors in the given radius: RADIUS
		neighbor_indexes = [rand_index] + retrieve_neighbors(in_pt=rand_pt.get_coordinates(), points_tree=points_tree,
		                                                     radius=RADIUS)
		remaining_neighbor_indexes = list(set(neighbor_indexes) - removed_points)
		neighbor_indexes = []
		neighbors = []
		for val in remaining_neighbor_indexes:
			if abs(data_points[val].angle - rand_pt.angle) <= ANGLE_TOLERANCE:
				neighbors.append(data_points[val])
				neighbor_indexes.append(val)

		# neighbors = data_points[remaining_neighbor_indexes]
		# Only consider neighbors that are supposed to be in the same heading direction.
		# neighbors = [nei for nei in data_points[remaining_neighbor_indexes] if abs(nei.angle - rand_pt.angle) <= ANGLE_TOLERANCE]

		# call find center method
		sample_pt = find_sample(rand_pt=rand_pt, neighbors=neighbors, draw_output=False)
		samples.append(sample_pt)

		# Remove elements
		removed_points = removed_points.union(neighbor_indexes)
		available_point_indexes = sorted(set(available_point_indexes) - removed_points)

	print 'NB SAMPLES: %s' % len(samples)
	samples_geojson = _to_geojson(samples)
	json.dump(samples_geojson, open('data/gps_data/gps_samples_07-11.geojson', 'w'))

	for s in samples:
		if s.lon > 50:
			plt.scatter(s.lon, s.lat)

	# Get x/y-axis in the same aspect
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()


def heading_vector_re_north(s, d):
	"""
	WRONG, doesn't account for the actual direction of the vector.!!!
	compute the vector angle from the north (north = 0 degree)
	:param s: source point
	:param d: destination point
	:return: angle in degrees from the north
	"""
	num = d.lat - s.lat
	den = d.lon - s.lon
	if den == 0 and num > 0:
		angle = 90
	elif den == 0 and num < 0:
		angle = -90
	else:
		angle = degrees(atan(num / den))
	return -1 * angle + 90

def vector_direction_re_north(s, d):
	"""
	Make the source as the reference of the plan. Then compute atan2 of the resulting destination point
	:param s: source point
	:param d: destination point
	:return: angle!
	"""

	# find the new coordinates of the destination point in a plan originated at source.
	new_d_lon = d.lon - s.lon
	new_d_lat = d.lat - s.lat
	 # angle = -angle + 90 is used to change the angle reference from east to north.
	angle = -degrees(atan2(new_d_lat, new_d_lon)) + 90

	# the following is required to move the degrees from -180, 180 to 0, 360
	if angle < 0:
		angle = angle + 360
	return angle


def get_paths(g):
	edges = {s:d for s,d in g.edges()}
	sources = [s for s in edges.keys() if g.in_degree(s) == 0]
	paths = []
	for source in sources:
		path = [source]
		s = source
		while (s in edges.keys()):
			path.append(edges[s])
			s = edges[s]
		paths.append(path)
	return paths

def road_segment_clustering(samples=None, minL=100, dist_threshold=0.001, angle_threshold=30):
	"""
	Create a graph from the samples. Nodes are samples, edges are road segments.
	:param samples:
	:param minL:
	:param dist_threshold:
	:param angle_threshold:
	:return:
	"""

	if samples is None:
		data = json.load(open('data/gps_data/gps_samples_07-11.geojson'))
		samples = _to_samplepoints(data)

	print 'NB nodes:', len(samples)
	# 1. Sort samples by weight, and create KD-Tree
	samples.sort(key=operator.attrgetter('weight'), reverse=True)
	simple_samples = [(s.lon, s.lat) for s in samples]
	samples_tree = cKDTree(simple_samples)

	# 2. for each element Si find Sp and Sq to form a smooth segment Sp-->Si-->Sq
	# node_degree = {_:0 for _ in range(len(samples))} # dictionary of node: degree
	g = nx.DiGraph()
	for ind, s in enumerate(samples):
		g.add_node(ind, speed=s.speed, angle=s.angle, lon=s.lon, lat=s.lat, weight=s.weight)

	for i, Si in enumerate(samples):
		print i, 'processing: ', Si
		if g.degree(i) > 2:
			print 'non treated'
			continue

		neighbors_index = [ind for ind in retrieve_neighbors(simple_samples[i], samples_tree, radius=dist_threshold)\
		             if samples[ind].lon != Si.lon or samples[ind].lat != Si.lat]
		neighbors = [samples[ind] for ind in neighbors_index]

		angle_pi_neighbors = []
		angle_iq_neighbors = []
		magnitude_v_neighbors = []
		haversine_v_neighbors = []
		for nei in neighbors:
			angle_pi_neighbors.append(vector_direction_re_north(nei, Si))
			angle_iq_neighbors.append(vector_direction_re_north(Si, nei))
			magnitude_v_neighbors.append(sqrt(pow((Si.lon - nei.lon), 2) + pow((Si.lat - nei.lat), 2)))
			haversine_v_neighbors.append(haversine((Si.lon, Si.lat), (nei.lon, nei.lat)))

		print 'Angle pi_nei:', angle_pi_neighbors
		# Find Sp
		Sp_candidates_index = [ind for ind, sp in enumerate(neighbors) if (abs(sp.angle - Si.angle) < angle_threshold) \
		                       and (abs(angle_pi_neighbors[ind] - Si.angle) < angle_threshold) \
		                       and (g.degree(neighbors_index[ind]) < 3)]
		Sp_candidates = [neighbors_index[ind] for ind in Sp_candidates_index]

		if len(Sp_candidates_index) > 0:
			scores = np.array(
					[magnitude_v_neighbors[ind] * abs(angle_pi_neighbors[ind] - Si.angle) * abs(Si.angle - neighbors[ind].angle) \
					 for ind in Sp_candidates_index])
			scores = np.array(
				[magnitude_v_neighbors[ind] for ind in Sp_candidates_index])
			Sp_index = Sp_candidates[np.argmin(scores)]
			g.add_edge(Sp_index, i)

		# Find Sq
		Sq_candidates_index = [ind for ind, sq in enumerate(neighbors) if (abs(sq.angle - Si.angle) < angle_threshold) \
		                       and (abs(angle_iq_neighbors[ind] - Si.angle) < angle_threshold) \
		                       and (g.degree(neighbors_index[ind]) < 3)]
		Sq_candidates = [neighbors_index[ind] for ind in Sq_candidates_index]

		if len(Sq_candidates_index) > 0:
			scores = np.array(
					[magnitude_v_neighbors[ind] * abs(angle_iq_neighbors[ind] - Si.angle) * abs(Si.angle - neighbors[ind].angle) \
					 for ind in Sq_candidates_index])
			#scores = np.array(
			#		[magnitude_v_neighbors[ind] for ind in Sq_candidates_index])
			Sq_index = Sq_candidates[np.argmin(scores)]
			g.add_edge(i, Sq_index)

	#segments = sorted(nx.weakly_connected_components(g), key=len, reverse=True)
	paths = get_paths(g)
	print 'NB edges:', len(g.edges()), 'NB segments:', len(paths)


	geojson = segments_to_geojson(paths, g)
	json.dump(geojson, open('data/gps_data/gps_segments_07-11.geojson', 'w'))

if __name__ == "__main__":
	# 1. find samples
	#traj_meanshift_sampling()

	# 2. find segments
	road_segment_clustering(minL=100, dist_threshold=0.0009, angle_threshold=30)






# INPUT_FILE_NAME = 'data/gps_data/gps_points_07-11.csv'
# RADIUS = 0.0001
# data_points, raw_points, points_tree = load_data(fname=INPUT_FILE_NAME)
# print 'nb points:',  len(data_points), 'points example:', data_points[0], raw_points[0]
#
# # input: point and angle:
# rand_pt = random.sample(data_points, 1)[0]
#
# # Find all neighbors in the given radius radius
# # neighbors = np.random.randint(10, size=(100, 2))
# neighbor_indexes = retrieve_neighbors(in_pt=rand_pt.get_coordinates(), points_tree=points_tree, radius=RADIUS)
# print 'pt:%s, angle:%s,  has %s neighbors' % (rand_pt.get_coordinates(), rand_pt.angle, len(neighbor_indexes))
# neighbors = data_points[neighbor_indexes]
#
# # call find center method
# sample_pt = find_center(rand_pt=rand_pt, neighbors=neighbors)
