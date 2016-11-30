
from collections import defaultdict
from geopy.distance import distance as geopy_distance
from geopy import Point
from geopy.distance import VincentyDistance


class MapGrid:
	def __init__(self, bbox=None, lon_step=None, lat_step=None):
		if bbox is not None:
			self.BBOX = bbox
		if lon_step is not None:
			self.lon_step = lon_step
		if lat_step is not None:
			self.lat_step = lat_step

		self.INDEX = None
		self.CELLS = self.generate_grid(bbox=self.BBOX, lon_step=self.lon_step, lat_step=self.lat_step)

	def create_cells_dictionary(self, cells):
		"""
		Creates a dictionary such as key is (lon_index, lat_index) and value is [min_lat, max_lat, min_lon, max_lon]
		:param cells: all cells generated in generate cell
		:return: dictionary.
		"""
		d_cells = defaultdict(list)
		for x in range(len(cells[0])):
			for y in range(len(cells)):
				d_cells[(x, y)] = cells[y][x]
		return d_cells

	@staticmethod
	def generate_cells(lst_latitude, lst_longitude):
		"""
		Generates a matrix of cells (array of array!)
		:param lst_latitude:
		:param lst_longitude:
		:return:
		"""
		lst = [[lst_latitude[i_lat-1], lst_latitude[i_lat], lst_longitude[j_lng-1], lst_longitude[j_lng]]\
			   for j_lng in range(1, len(lst_longitude)) for i_lat in range(1, len(lst_latitude))]

		cells = {}
		for i, element in enumerate(lst):
			cells[i] = element
		return cells

	@staticmethod
	def generate_cells_matrix(lst_latitude, lst_longitude):
		"""
		Generates a matrix of cells.
		:param lst_latitude: list of latitude chunks
		:param lst_longitude: list of longitude chunks
		:return: list of list.
		"""
		cells_matrix = []
		for y_lat in range(1, len(lst_latitude)):
			vec = []
			for x_lon in range(1, len(lst_longitude)):
				vec.append([lst_latitude[y_lat-1], lst_latitude[y_lat], lst_longitude[x_lon-1], lst_longitude[x_lon]])
			cells_matrix.append(vec)
		return cells_matrix

	@staticmethod
	def generate_cells_dictionary(lst_latitude, lst_longitude):
		"""
		Generates a dictionary of cells: key (lon_index, lat_index) and value is [min_lat, max_lat, min_lon, max_lon]
		:param lst_latitude: list of latitude_chunks
		:param lst_longitude: list of longitue_chunks
		:return: dictionary.
		"""
		d_cell = defaultdict(list)
		for y_lat in range(1, len(lst_latitude)):
			for x_lon in range(1, len(lst_longitude)):
				d_cell[(x_lon-1, y_lat-1)] = [lst_latitude[y_lat-1], lst_latitude[y_lat], lst_longitude[x_lon-1], lst_longitude[x_lon]]
		return d_cell

	def generate_grid(self, bbox, lon_step=1, lat_step=1, output='cd'):
		"""
		Divide a bounding box into cells of the same size with x and y as side distances in km.
		:param bbox: expected in the format: minLat, maxLat, minLng, maxLng
		:param lon_step: the size of the horizental side
		:param lat_step: the size of the vertical side
		:param output: cc for cell centers i.e., (lat, lng) of each cell, c for cells, cm for cell matrix,
		cd for cell dictionary.
		:return: dictionary of cells and the index to be used to find the cell that contains a point.
		"""

		bearingEast = 90
		bearingNorth = 0
		minLatitude, maxLatitude, minLongitude, maxLongitude = bbox
		p_sw = Point(minLatitude, minLongitude)
		p_se = Point(minLatitude, maxLongitude)
		p_ne = Point(maxLatitude, maxLongitude)
		lon_span_km = geopy_distance(p_sw, p_se).kilometers
		lat_span_km = geopy_distance(p_ne, p_se).kilometers
		lat_chunks = range(int(lat_span_km / lon_step) + 1)
		lon_chunks = range(int(lon_span_km / lat_step) + 1)

		# return center cells
		if output == 'cc':
			lst_lat = [(VincentyDistance(kilometers=lon_step * x + lon_step / 2).destination(p_sw, bearingNorth)).latitude for x in lat_chunks]
			lst_lon = [(VincentyDistance(kilometers=lat_step * y + lat_step / 2).destination(p_sw, bearingEast)).longitude for y in lon_chunks]
			self.INDEX = {'lats': lst_lat, 'lons': lst_lon}

			return [(x, y) for x in lst_lat[:-1] for y in lst_lon[:-1]]

		# return list of cells
		elif output == 'c':
			lst_lat = [(VincentyDistance(kilometers=lon_step * x + lon_step).destination(p_sw, bearingNorth)).latitude for x in lat_chunks]
			lst_lon = [(VincentyDistance(kilometers=lat_step * y + lat_step).destination(p_sw, bearingEast)).longitude for y in lon_chunks]
			cells = self.generate_cells(lst_lat, lst_lon)
			self.INDEX = {'lats': lst_lat, 'lons': lst_lon}
			return cells

		# return cells matrix
		elif output == 'cm':
			lst_lat = [minLatitude]+[(VincentyDistance(kilometers=lat_step * y + lat_step).destination(p_sw, bearingNorth)).latitude for y in lat_chunks]
			lst_lon = [minLongitude]+[(VincentyDistance(kilometers=lon_step * x + lon_step).destination(p_sw, bearingEast)).longitude for x in lon_chunks]
			self.INDEX = {'lats': lst_lat, 'lons': lst_lon}
			cells = self.generate_cells_matrix(lst_lat, lst_lon)
			return cells

		# return cells dictionary:
		elif output == 'cd':
			lst_lat = [minLatitude]+[(VincentyDistance(kilometers=lat_step * y + lat_step).destination(p_sw, bearingNorth)).latitude for y in lat_chunks]
			lst_lon = [minLongitude]+[(VincentyDistance(kilometers=lon_step * x + lon_step).destination(p_sw, bearingEast)).longitude for x in lon_chunks]
			self.INDEX = {'lats': lst_lat, 'lons': lst_lon}
			cells = self.generate_cells_dictionary(lst_lat, lst_lon)
			return cells


	def find_cell(self, point):
		"""
		Find the cell that contains a point given as input.
		:param self:
		:param point: geo coordinates of a point in this format: [lon, lat]
		:return: None if the point is outside the current map bounding box, the id of the cell (i, j) otherwise.
		"""
		# point: [lon, lat]
		if point[0] < self.INDEX['lons'][0] or point[0] > self.INDEX['lons'][-1] or \
			point[1] < self.INDEX['lats'][0] or point[1] > self.INDEX['lats'][-1]:
			return None
		i = next(indx for indx, val in enumerate(self.INDEX['lons']) if val > point[0])
		j = next(indx for indx, val in enumerate(self.INDEX['lats']) if val > point[1])
		return (i-1, j-1)

	def get_cell_center(self, cell_id):
		return [(self.CELLS[cell_id][2]+self.CELLS[cell_id][3]) / 2, (self.CELLS[cell_id][0]+self.CELLS[cell_id][1]) / 2]


if __name__ == '__main__':
	# test the class

	# Initialize the parameters
	doha_bbox = [25.21558, 25.387415, 51.44717, 51.630657]
	lon_step = 3
	lat_step = 3

	# instantiate the class
	grid = MapGrid(bbox=doha_bbox, lon_step=lon_step, lat_step=lat_step)
	print 'CELLS: ', grid.CELLS
	point = [51.48, 25.30]
	cell = grid.find_cell(point=point)
	print 'point:', point
	print 'container cell:', cell, grid.CELLS[cell]



