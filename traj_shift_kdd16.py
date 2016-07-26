import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from math import tan, radians

BIN_SEGMENT_LENGTH = 2  # length of bins in meters


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

if __name__ == "__main__":
	# input: point and angle:
	in_pt = [3, 5]
	angle = 30
	# prepare some input neighbors
	neighbors = list()
	px = [3, 6, 6, 4, 5, 2, 7, 7]
	py = [2, 2, 7, 1, 2, 4, 3, 2]
	for pt in zip(px, py):
		neighbors.append(pt)

	sample_pt = find_sample(in_pt=in_pt, angle=angle, neighbors=neighbors)
