import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from math import tan, sin, cos, asin, radians, exp, sqrt, ceil
from scipy.spatial import cKDTree
import datetime
import random

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

class line:
	def __init__(self, slope=None, intercept=None):
		self.slope = slope if slope is not None else None
		self.intercept = intercept if intercept is not None else None

	def perpendecular(self, at_pt=None):
		slope = -1 / self.slope
		intercept = at_pt[1] - slope * at_pt[0] if at_pt is not None else 0
		return line(slope=slope, intercept=intercept)

	def plot(self, color=None, width=2, pt=None):
		x = np.array([-0.00002, -0.0001, 0, 0.00001, 0.00002])+pt[0]
		if color is not None:
			plt.plot(x, self.intercept + self.slope * x, linewidth=width, color=color)
		else:
			plt.plot(x, self.intercept + self.slope * x, linewidth=width)

	def __repr__(self):
		return 'y = %sx + %s' % (self.slope, self.intercept)

def project_point_into_line(pt, parallel_line, perpendecular_line):
	"""
    This functions finds the projected point (ppt) of a point (pt) on a line (line). It computed the intersection
    between line at the perpondecular line to it that goes through pt. 
    :param pt: point (x, y)
    :param line: line defined with (intercept, slope)
    :returns: intersection point (x, y)
    """
	# 1. find the line that is parallel to parallel line that goes thu pt.
	intercept = pt[1] - parallel_line.slope * pt[0]
	#line(parallel_line.slope, intercept).plot(color='green', width=1)

	# 2. find intersection point between pt_line and perpendicular_line
	X = np.array([[1, - parallel_line.slope], [1, -perpendecular_line.slope]])
	Y = np.array([intercept, perpendecular_line.intercept])
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


def find_sample(rand_pt=None, neighbors=None):
	in_pt = rand_pt.get_coordinates()
	angle = rand_pt.angle

	# equation of the heading line of the point:
	eq_heading = line_of_gps_point(in_pt, angle)
	# equation of the perpendicular line to the point's heading line:
	eq_perpend = eq_heading.perpendecular(at_pt=in_pt)
	print 'heading eq:', eq_heading
	print 'perpendecular eq:', eq_perpend

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
	LANE_WIDTH = 3.3 # lane width in meters
	segment_length = max([haversine(neighbors[i].get_coordinates(), neighbors[j].get_coordinates())\
	                      for i in range(len(neighbors)) for j in range(i+1, len(neighbors))])
	nb_bins = int(ceil(segment_length/LANE_WIDTH))
	print 'nb_bins: ', nb_bins
	histog = np.histogram(axis, bins=nb_bins, density=True)
	densities, bin_limits = histog[0], histog[1]

	# Find the relevant/central bin:
	common_denominator = sum([distance_heading_speed(rand_pt, nei) for nei in neighbors])
	bin_votes = []
	for i, (bmin, bmax) in enumerate(zip(bin_limits[:-1], bin_limits[1:])):
		# find neighbors that are inside the bin limits
		votes = 0
		for nei in neighbors:
			if (axis_label == 'x-axis' and nei.lon >= bmin and nei.lon < bmax) or \
					(axis_label == 'y-axis' and nei.lat >= bmin and nei.lat < bmax):
				votes += distance_heading_speed(rand_pt, nei)
		bin_votes.append(votes/common_denominator)
	max_density_bin = np.argmax(bin_votes)
	print 'max bin: %s, bins votes: %s' % (max_density_bin, bin_votes)
	#max_density_bin = np.argmax(densities)
	marker = (bin_limits[max_density_bin] + bin_limits[max_density_bin + 1]) / 2
	# find the other component of the sample point: parallel to x-axis or y-axis
	if axis_label == 'x-axis':
		Sx = marker
		Sy = eq_perpend.intercept + eq_perpend.slope * Sx
	else:
		Sy = marker
		Sx = (Sy - eq_perpend.intercept) / eq_perpend.slope

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
	plt.xlim([minx-0.00001, maxx+0.00001])
	plt.ylim([miny-0.00001, maxy+0.00001])
	# Get x/y-axis in the same aspect
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()

	return (Sx, Sy)

def retrieve_neighbors(in_pt, points_tree, radius=5):
	"""
	Retrieve all points within a radius to an input point in_pt
	:param in_pt: intput point
	:param points_tree: KDTree of points, needs to be precomputed
	:param radius: distance radius
	:return: list of indexes of neighbors.
	"""

	#points = np.random.randint(50, size=(100, 2))
	#points_tree = cKDTree(points)
	neighbors = points_tree.query_ball_point(x=in_pt, r=radius, p=2)
	return neighbors


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


if __name__ == "__main__":
	INPUT_FILE_NAME = 'data/gps_data/gps_points_07-11.csv'
	data_points, raw_points, points_tree = load_data(fname=INPUT_FILE_NAME)
	print 'nb points:',  len(data_points), len(raw_points)
	print 'points example:', data_points[0:3], raw_points[0:3]

	# input: point and angle:
	rand_pt = random.sample(data_points, 1)[0]
	pt_coordinates = rand_pt.get_coordinates()
	angle = rand_pt.angle

	# prepare some input neighbors
	# neighbors = np.random.randint(10, size=(100, 2))

	neighbor_indexes = retrieve_neighbors(in_pt=pt_coordinates, points_tree=points_tree, radius=0.0001)
	print 'pt:%s, angle:%s,  has %s neighbors' % (pt_coordinates, angle, len(neighbor_indexes))
	neighbors = data_points[neighbor_indexes]

	# call find sample method
	sample_pt = find_sample(rand_pt=rand_pt, neighbors=neighbors)
