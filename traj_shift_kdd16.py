import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from math import tan, radians
from scipy.spatial import cKDTree
import datetime

BIN_SEGMENT_LENGTH = 2  # length of bins in meters

class GpsPoint:
	def __init__(self, data=None):
		if data is not None:
			self.btid = data[4]
			self.speed = float(data[5])
			self.timestamp = datetime.datetime.strptime(data[6], '%Y-%m-%d %H:%M:%S+03')
			self.lon = data[7]
			self.lat = data[8]
			self.angle = data[9]

	def get_coordinates(self):
		"""
		return the lon,lat of a gps point
		:return: tuple (lon, lat)
		"""
		return (self.lon, self.lat)

	def __str__(self):
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

	def plot(self, color=None, width=2):
		x = np.arange(-20, 20)
		if color is not None:
			plt.plot(x, self.intercept + self.slope * x, linewidth=width, color=color)
		else:
			plt.plot(x, self.intercept + self.slope * x, linewidth=width)


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
	line(parallel_line.slope, intercept).plot(color='green', width=1)

	# 2. find intersection point between pt_line and perpendecular_line
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


def find_sample(in_pt, angle, neighbors):
	# equation of the heading line of the point:
	eq_heading = line_of_gps_point(in_pt, angle)

	# equation of the perpondecular line to the point's heading line:
	eq_perpend = eq_heading.perpendecular(at_pt=in_pt)

	# Project all points into the perpendecular line
	projected_points = list()
	for pt in neighbors:
		intersec_pt = project_point_into_line(pt, eq_heading, eq_perpend)
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

	histog = np.histogram(axis, bins=3, density=True)
	densities, bins = histog[0], histog[1]
	max_density_bin = np.argmax(densities)
	marker = (bins[max_density_bin] + bins[max_density_bin + 1]) / 2
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
	eq_heading.plot(color='blue')
	eq_perpend.plot(color='red')
	plt.scatter(in_pt[0], in_pt[1], marker='8', s=100)
	for pt, intersec_pt in zip(neighbors, projected_points):
		plt.scatter(intersec_pt[0], intersec_pt[1], marker='*', s=50)
		plt.scatter(pt[0], pt[1])
	plt.scatter(Sx, Sy, marker='D', s=50)
	plt.xlim([-2, 10])
	plt.ylim([-2, 10])
	# Get x/y-axis in the same aspect
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()

	return (Sx, Sy)


def nearest_neighbors(in_pt, points_tree, radius=5):
	"""
	Retrieve all points within a radius to an input point in_pt
	:param in_pt: intput point
	:param points_tree: KDTree of points, needs to be precomputed
	:param radius: distance radius
	:return: list of indexes of neighbors.
	"""

	points = np.random.randint(50, size=(100, 2))
	points_tree = cKDTree(points)
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
	return data_points, raw_points, points_tree


if __name__ == "__main__":
	INPUT_FILE_NAME = 'data/gps_data/gps_points.csv'
	data_points, raw_points, points_tree = load_data(fname=INPUT_FILE_NAME)
	print len(data_points), len(raw_points)
	print data_points[0:3], raw_points[0:3]

	# input: point and angle:
	in_pt = [3, 5]
	angle = 30

	# prepare some input neighbors
	neighbors = np.random.randint(10, size=(100, 2))

	# call find sample method
	sample_pt = find_sample(in_pt=in_pt, angle=angle, neighbors=neighbors)
