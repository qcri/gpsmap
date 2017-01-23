from collections import defaultdict, Counter
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
import fiona
import random


BIN_SEGMENT_LENGTH = 2  # length of bins in meters


class GpsPoint:
	def __init__(self, data=None):
		if data is not None:
			self.btid = int(data[4])
			self.speed = float(data[5])
			self.timestamp = datetime.datetime.strptime(data[6], '%Y-%m-%d %H:%M:%S+03')
			self.lon = float(data[7])
			self.lat = float(data[8])
			self.angle = float(data[9])
			self.ptid = int(data[2])
			self.locid = int(data[3])

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
	def __init__(self, spid=None, speed=None, lon=None, lat=None, angle=None, weight=None):
		self.speed = speed
		self.lon = lon
		self.lat = lat
		self.angle = angle
		self.weight = weight
		self.spid = int(spid)

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
			# Thus, I'm giving a special interpretation to slope and intercept here to capture this case.
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


def to_geojson(samples):
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
		feature['properties']['id'] = s.spid
		geojson['features'].append(feature)
	return geojson


def segments_to_geojson(segments, g):
	"""
	Generate the geojson object of a list of segments
	:param samples: list of segments
	:return: one geojson object that contains all the points.
	"""
	geojson = {'type': 'FeatureCollection', 'features': []}
	for segment_id, s in enumerate(segments):
		feature = {'type': 'Feature', 'properties': {}, 'geometry': {'type': 'LineString', 'coordinates': []}}
		coordinates = [[g.node[p]['lon'], g.node[p]['lat']] for p in list(s)]
		pt_ids = [g.node[p]['id'] for p in list(s)]
		feature['geometry']['coordinates'] = coordinates
		feature['properties'] = {'segment_id': segment_id, 'pt_ids': pt_ids}
		geojson['features'].append(feature)
	return geojson

def links_to_geojson(links, samples, max_link_length):
	"""
	Generate the geojson object of a list of links between segments
	:param samples: list of samples
	:return: one geojson object that contains all the points.
	"""
	link_cnt = Counter(links)
	geojson = {'type': 'FeatureCollection', 'features': []}
	link_id = 0
	for link, w in link_cnt.most_common(len(link_cnt)):
		feature = {'type': 'Feature', 'properties': {}, 'geometry': {'type': 'LineString', 'coordinates': []}}
		coordinates = [[samples[link[0]].lon, samples[link[0]].lat], [samples[link[1]].lon, samples[link[1]].lat]]
		if haversine(coordinates[0], coordinates[1]) > max_link_length:
			continue
		pt_ids = [samples[link[0]].spid, samples[link[1]].spid]
		feature['geometry']['coordinates'] = coordinates
		feature['properties'] = {'link_id': link_id, 'weight': w, 'pt_ids': pt_ids}
		geojson['features'].append(feature)
		link_id += 1
	return geojson

def to_segments(geojson):
	"""
	return segment coordinates and segment ids. Both are list of lists. Each internal list is a segment.
	:param geojson:
	:return:
	"""
	segment_pt_ids = defaultdict(list)
	segment_pt_coordinates = defaultdict(list)
	for s in geojson['features']:
		segment_id = s['properties']['segment_id']
		segment_pt_coordinates[segment_id].append(s['geometry']['coordinates'])
		segment_pt_ids[segment_id] =s['properties']['pt_ids']

	return segment_pt_coordinates, segment_pt_ids


def to_samplepoints(geojson):
	samples = []
	samples_dict = {}
	for s in geojson['features']:
		if s['geometry']['coordinates'][0] == 0 or s['geometry']['coordinates'][1] == 0: continue
		samples.append(SamplePoint(spid=int(s['properties']['id']), speed=s['properties']['speed'],
		                           angle=s['properties']['angle'], weight=s['properties']['weight'],
		                           lon=s['geometry']['coordinates'][0], lat=s['geometry']['coordinates'][1]))

		samples_dict[int(s['properties']['id'])] = SamplePoint(spid=int(s['properties']['id']), speed=s['properties']['speed'],
		                           angle=s['properties']['angle'], weight=s['properties']['weight'],
		                           lon=s['geometry']['coordinates'][0], lat=s['geometry']['coordinates'][1])
	return samples, samples_dict


def uniquify(seq):
	"""
	Uniquify a list of items, eg. [1,1,1,2,2,2,3,3,4] => [1,2,3,4]
	:param seq: list of items
	:return: unique values is seq, with order preserved!
	"""
	if len(seq) < 2:
		return seq
	return [seq[0]] + [x for i, x in enumerate(seq[1:]) if x != seq[i]]
	# seen = set()
	# seen_add = seen.add
	# return [x for x in seq if not (x in seen or seen_add(x))]


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
	"""
	Compute all paths + consider single nodes!
	:param g:
	:return:
	"""
	edges = {s: d for s, d in g.edges()}
	sources = [s for s in edges.keys() if g.in_degree(s) == 0]
	paths = []
	for source in sources:
		# handle the case of reflexive links
		path = [source]
		s = source
		while (s in edges.keys()):
			path.append(edges[s])
			s = edges[s]
		paths.append(path)
	path_nodes = [n for path in paths for n in path]
	for node in g.nodes():
		if node not in path_nodes:
			paths.append([node])
	return paths


def get_paths_with_reflexive_links(g):
	"""
	Compute all paths + consider single nodes!
	:param g:
	:return:
	"""
	edges = {s: d for s, d in g.edges()}
	sources = [s for s in edges.keys() if g.in_degree(s) == 0]
	paths = []
	for source in sources:
		# handle the case of reflexive links
		path = [source]
		s = source
		while (s in edges.keys()):
			path.append(edges[s])
			s = edges[s]
		paths.append(path)
	path_nodes = [n for path in paths for n in path]
	for node in g.nodes():
		if node not in path_nodes:
			paths.append([node])
	return paths


def create_trajectories(INPUT_FILE_NAME='data/gps_data/gps_points_07-11.csv', waiting_threshold=5):
	"""
	return all trajectories.
	The heuristic is simple. Consider each users sorted traces not broken by more than 1 hour as trajectories.
	:param waiting_threshold: threshold for trajectory split expressed in seconds.
	:return: list of lists of trajectories
	"""

	data_points, raw_points, points_tree = load_data(fname=INPUT_FILE_NAME)
	detections = defaultdict(list)
	for p in data_points:
		detections[p.btid].append(p)

	# compute trajectories: split detections by waiting_threshold
	print 'Computing trajectories'
	trajectories = []
	for btd, ldetections in detections.iteritems():
		points = sorted(ldetections, key=operator.attrgetter('timestamp'))
		source = 0
		destination = 0
		for i in range(1, len(points)):
			delta = points[i].timestamp - points[source].timestamp
			if delta.days * 24 * 3600 + delta.seconds > waiting_threshold:
				trajectories.append(points[source: i])
				source = i + 1
				i += 1
		if source < len(points):
			trajectories.append(points[source: -1])
	return trajectories

def assign_points_to_cells_v1(mgrid, points):
	"""
	Assigns point to cells.
	:param mgrid: this is an instance of MapGrid
	:param points: dictionary of points
	:return: two dictionaries. One that maps points to cells, and one that map cells to points.
	"""
	# initialize the two dictionaries
	cell_to_points = defaultdict(list) # cell --> [checkin ids, ]
	point_to_cell = dict() # checkin_id --> cell_id

	for ch_id, point in points.iteritems():
		point_coordinates = point['geo'] # assumes tuple (lon, lat)
		cell_id = mgrid.find_cell(point_coordinates)
		if cell_id is None:
			continue
		point_to_cell[ch_id] = cell_id
		cell_to_points[cell_id].append(ch_id)
	return cell_to_points, point_to_cell

def read_points(fname):
	"""
	Read points from file.
	:param fname: file name
	:return: dictionaty of points.
	"""
	points = defaultdict(dict)
	with open(fname, 'r') as f:
		for line in f:
			X, Y, uid, location_i, device_id, speed, timestamp, longitude, latitude, angle = line.strip().split('\t')
			points[uid] = {'timestamp': timestamp,
			               'geo': (float(longitude), float(latitude)),
			               'speed': float(speed),
			               'device_id': device_id,
			               'angle': angle}
	return points

def is_point_in_bbox(point, bbox):
	"""
	Checks whether a point (lon, lat) is inside a bounding box
	:param point: lon, lat
	:param bbox: minlat, maxlat, minlon, maxlon
	:return: True or False
	"""
	if  bbox[3] >= point[0] >= bbox[2] and bbox[1] >= point[1] >= bbox[0]:
		return True
	return False


def build_road_network_from_shapefile_with_no_middle_nodes(shape_file, city=None, bounding_polygon=None):
	"""
	IMPORTANT: I took it from my other project: cityres2
	This function builds a road graph of a city from its road shapefile.
	The idea is to create an edge for each consecutive nodes in each path.
	 Use fiona to read the shape file.
	:param shape_file: the road shape file of the city
	:param city: the name of the city
	:param bounding_polygon: personalized bbox of some cities. Helpful to capture part of what is in the shapefile
	:param simplify: True => behave like nx.read_shp (ignore middle nodes), False => create full graph
	:return: a graph
	"""

	g = nx.DiGraph()
	#road_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']
	sh = fiona.open(shape_file)
	intersection_nodes = set()
	all_nodes = list()
	paths = []
	if bounding_polygon is None:
		# 1. get all nodes

		for obj in sh:
			path = obj['geometry']['coordinates']
			#if obj['properties']['type'] not in road_types:
			#	continue
			if len(path) < 2:
				continue
			if all(isinstance(elem, list) for elem in path):
				# I have cases where path is a list of lists (list of paths)
				for elem in path:
					paths.append(elem)
					intersection_nodes.add(elem[0])
					intersection_nodes.add(elem[-1])
					all_nodes += elem
			else:
				paths.append(path)
				intersection_nodes.add(path[0])
				intersection_nodes.add(path[-1])
				all_nodes += path

		# 2. compute node frequencies and remove
		nodes_cnt = Counter(all_nodes)
		inters_nodes = [node for node, freq in nodes_cnt if freq > 1]
		intersection_nodes = intersection_nodes.union(inters_nodes)

		# 3. Build the road network: tested and works as expected :)
		for path in paths:
			for i in range(len(path)-1):
				if path[i] not in intersection_nodes:
					continue
				source = path[i]
				for j in range(i+1, len(path)):
					if path[j] in intersection_nodes and path[j] != source:
						target = path[j]
						g.add_edge(source, target)
						i = j
						break
	else:
		# 1. get all nodes
		for obj in sh:
			path = obj['geometry']['coordinates']
			if len(path) < 2:
				continue
			if is_point_in_bbox(path[0], bounding_polygon):
				intersection_nodes.add(path[0])
			if is_point_in_bbox(path[-1], bounding_polygon):
				intersection_nodes.add(path[-1])

			for i in range(len(path)):
				all_nodes.append(path[i])
			paths.append(path)
		# 2. compute node frequencies and remove
		nodes_cnt = Counter(all_nodes)
		inters_nodes = [node for node, freq in nodes_cnt.iteritems() if freq > 1 and is_point_in_bbox(node, bounding_polygon)]
		intersection_nodes = intersection_nodes.union(inters_nodes)

		# 3. Build the road network: tested and works as expected :)
		for path in paths:
			for i in range(len(path)-1):
				if path[i] not in intersection_nodes:
					continue
				source = path[i]
				for j in range(i+1, len(path)):
					if path[j] in intersection_nodes and path[j] != source:
						target = path[j]
						g.add_edge(source, target)
						i = j
						break
	return g

def  remove_segments_with_no_points(g, points):
	"""
	This method should clean a road network g by removing all edges for which there are no points within a certain
	distance. i.e., segments with no data.
	:param g:
	:param points:
	:return:
	"""

	return g

def get_marble_holes(rnet=None, radius=1000, frequency=5, type='holes', starting_point=None):
	"""
	Compile a list of holes or marbles within a radius, each frequency distance.

	:param rnet: the road network
	:param radius: the max distance from the starting_point, in meters
	:param frequency: create a hole/marble every x meters
	:param type: 'holes' (start from a random point), 'marble' (start from a given point)
	:return: list of points.
	"""

	if type == 'holes':
		# generate a random position ==> lets pick a random node in the network.
		nodes = rnet.nodes()
		starting_point = nodes[random.randint(0, len(nodes) - 1)]

	visited_nei = []
	all_nei = rnet.edges(starting_point)
	relevant_nei = set([nei[1] for nei in all_nei if haversine(starting_point, nei[1]) <= radius])
	process_nei = relevant_nei.copy()
	while True:
		# exit the loop when there are no more relevant point to process
		if len(process_nei) == 0:
			break
		nei = process_nei.pop()
		nei_neighbors = [n[1] for n in rnet.edges(nei) if haversine(starting_point, n[1]) <= radius]
		relevant_nei.union(set(nei_neighbors))
		for n in nei_neighbors:
			if n not in relevant_nei:
				process_nei.append(n)

	# TODO: to be continued









