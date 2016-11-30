import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
import json
import numpy as np
from pyproj import Proj, transform
from matplotlib.collections import PatchCollection
from shapely.geometry import Point, MultiPoint, MultiPolygon, shape, Polygon
from descartes import PolygonPatch



mpl.rcParams['axes.facecolor'] = 'gray'
qatar_bbox = [24.4708, 26.185, 50.7321, 51.7017]
doha_bbox = [25.21558, 25.387415, 51.44717, 51.630657]


def esri_to_gps(x, y):
	inProj = Proj(init='epsg:28600')
	outProj = Proj(init='epsg:4326')
	x2,y2 = transform(inProj, outProj, x, y)
	return (x2, y2)


def draw_polygon(bmap, polygon):
	"""
	Draw polygon
	:param bmap: basemap instance
	:param polygon: list of points each of which has [lon, lat]
	:return:
	"""
	X, Y = [], []
	for point in polygon:
		X.append(point[0])
		Y.append(point[1])
	x, y = bmap(X, Y)
	bmap.plot(x, y, color='red', linewidth=2)

def draw_point(bmap, point, color='gold', size=100):
	"""
	Draws a point on the map.
	:param bmap:
	:param point:
	:return:
	"""
	X, Y = [point[0]], [point[1]]
	x, y = bmap(X, Y)
	bmap.scatter(x, y, s=size, c=color, marker="*", cmap=cm.jet, alpha=1.0, edgecolors='face', linewidths=0.5)

def draw_cell(bmap, cell_bbx, color='white'):
	"""
	Draw a cell
	:param bmap: basemap instance
	:param cell_bbx: [min_lat, max_lat, min_lon, max_lon]
	:return:
	"""
	X, Y = [], []
	# NE: max_lon, max_lat
	X.append(cell_bbx[3])
	Y.append(cell_bbx[1])
	# NW: min_lon, max_lat
	X.append(cell_bbx[2])
	Y.append(cell_bbx[1])
	# SW: min_lon, min_lat
	X.append(cell_bbx[2])
	Y.append(cell_bbx[0])
	# SE: max_lon, min_lat
	X.append(cell_bbx[3])
	Y.append(cell_bbx[0])
	# complete the cercle, go back to NE:
	X.append(cell_bbx[3])
	Y.append(cell_bbx[1])
	x, y = bmap(X, Y)
	bmap.plot(x, y, color=color, linewidth=2)

def draw_cell_center(bmap, cell_bbx=None, color='red', size=10, rank=None):
	X, Y = [], []
	X.append(cell_bbx[2]+(cell_bbx[3]-cell_bbx[2])/2)
	Y.append(cell_bbx[0]+(cell_bbx[1]-cell_bbx[0])/2)
	x, y = bmap(X, Y)
	bmap.scatter(x, y, s=size, c=color, marker=".", cmap=cm.jet, alpha=1.0, edgecolors='face', linewidths=0)
	#plt.text(x[0], y[0], str(rank), color=color, fontsize=8)


def init_doha_map(bbox=None, shapefile=None, ax=None):
	# initialization of basemap
	#plt.figure(figsize=[18, 18])
	if ax is None:
		plt.figure(figsize=[18, 18])
		bmap = Basemap(llcrnrlon=bbox[2] - 0.002, llcrnrlat=bbox[0] -0.002, urcrnrlon=bbox[3] + 0.028, urcrnrlat=bbox[1] + 0.02, resolution='h', \
				projection='merc', lon_0=bbox[2] + (bbox[3] - bbox[2]) / 2, lat_0=bbox[0] + (bbox[1] - bbox[0]) / 2, area_thresh=1000.)
	else:
		bmap = Basemap(llcrnrlon=bbox[2], llcrnrlat=bbox[0], urcrnrlon=bbox[3] + 0.006, urcrnrlat=bbox[1] + 0.009, resolution='h', \
					projection='merc', lon_0=bbox[2] + (bbox[3] - bbox[2]) / 2, lat_0=bbox[0] + (bbox[1] - bbox[0]) / 2, area_thresh=1000., ax=ax)

	bmap.drawmapboundary(fill_color='white')# fill to edge
	bmap.drawcountries()
	bmap.drawstates(color='white', linewidth=1.)
	bmap.drawmapboundary(fill_color='aqua')
	bmap.fillcontinents(color='white', lake_color='gray', zorder=0)
	if shapefile is not None:
		#print 'location:', shapefile
		bmap.readshapefile(shapefile, 'Qatar', linewidth=0.1, color='gray')
	return bmap

def draw_map(bbx=None):
	"""
	Plot a map using basemap + shapefiles.
	:param bbx: bounding boxes of the location to plot: minLat, maxLat, minLon, maxLon
	:return:
	"""
	# initialization of basemap
	plt.figure(figsize=[18, 18])
	m = Basemap(llcrnrlon=bbx[2], llcrnrlat=bbx[0], urcrnrlon=bbx[3]+0.006, urcrnrlat=bbx[1]+0.009, resolution='i',\
				projection='merc', lon_0=bbx[2]+(bbx[3]-bbx[2])/2, lat_0=bbx[0]+(bbx[1]-bbx[0])/2, area_thresh=1000.)

	m.drawmapboundary(fill_color='white')# fill to edge
	m.drawcountries()
	m.drawstates(color='white', linewidth=1.)
	m.drawmapboundary(fill_color='aqua')
	m.fillcontinents(color='white', lake_color='gray', zorder=0)

	#m.readshapefile('../data/doha_qatar.osm/doha_qatar_osm_polygon', 'doha')
	#m.readshapefile('../data/qgis/qgis', 'doha')


#draw_map(doha_bbox)
#plt.show()

def draw_road_network():
	"""
	Draws the road network from the road geojson file. Use simply matplotlib
	:return:
	"""
	with open('../data/doha_qatar.imposm-geojson/doha_qatar_roads_gen1.geojson', 'r') as f:
		roads = json.load(f)['features']
		types = set()
		for road in roads:
			type = road['properties']['type']
			types.add(type)
			coords = road['geometry']['coordinates']
			x  = list()
			y = list()
			for point in coords:
				x.append(point[0])
				y.append(point[1])
			plt.plot(x, y, color='gold', linewidth=1)
	#plt.show()
	plt.savefig('../figs/doha_road_network.png', format='PNG', bbox_inches='tight')
#draw_road_network()

def draw_road_network_from_shapefile(shape_file):
	"""
	Draws the road network from the road geojson file. Use simply matplotlib
	:return:
	"""
	mpl.rcParams['axes.facecolor'] = 'gray'

	road_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']
	sh = fiona.open(shape_file)
	for obj in sh:
		path = obj['geometry']['coordinates']
		if obj['properties']['type'] not in road_types:
			continue
		x, y = [], []
		for point in path:
			x.append(point[0])
			y.append(point[1])
		plt.plot(x, y, color='gold', linewidth=1)
	plt.axis('equal')

	plt.show()

def draw_doha_shapefiles():
	bmap = init_doha_map(bbox=doha_bbox)
	# draw different qgis in different colors:
	with open('../data/doha_qatar.imposm-geojson/doha_qatar_admin.geojson', 'r') as f:
		admins = json.load(f)['features']
		for admin in admins:
			if admin['properties']['admin_leve'] == 10.0 and  len(admin['geometry']['coordinates']) == 1:
				polygon = admin['geometry']['coordinates'][0]
				draw_polygon(bmap, polygon=polygon)

#draw_doha_shapefiles()


def draw_doha_qgis_zones(fname, att_name='rings'):
	bmap = init_doha_map(bbox=doha_bbox)
	# draw different qgis in different colors:
	cnt = 0
	with open(fname, 'r') as f:
		admins = json.load(f)['features']
		for admin in admins:
			polygon = [esri_to_gps(p[0], p[1]) for p in admin['geometry'][att_name][0]]
			#polygon = admin['geometry'][att_name][0]
			print cnt, polygon
			draw_polygon(bmap, polygon=polygon)
			cnt += 1
	plt.show()
fname = '../data/qgis/zones_crs28600.json'
#draw_doha_qgis_zones(fname, att_name='rings')


def draw_dict_cells(bmap=None, dcells=None, bbox=None, cell_activation_rank=None, color='white', lon_step=0.2):
	if bmap is None:
		# initialization of basemap
		plt.figure(figsize=[18, 18])
		bmap = Basemap(llcrnrlon=bbox[2] - 0.002, llcrnrlat=bbox[0] -0.002, urcrnrlon=bbox[3] + 0.028, urcrnrlat=bbox[1] + 0.02, resolution='h', \
					projection='merc', lon_0=bbox[2] + (bbox[3] - bbox[2]) / 2, lat_0=bbox[0] + (bbox[1] - bbox[0]) / 2, area_thresh=1000.)

		bmap.drawmapboundary(fill_color='white')# fill to edge
		bmap.drawcountries()
		bmap.drawstates(color='white', linewidth=1.)
		bmap.drawmapboundary(fill_color='aqua')
		bmap.fillcontinents(color='gray', lake_color='gray', zorder=0)
		bmap.readshapefile('../data/doha_qatar.osm/doha_qatar_osm_polygon', 'doha')

	for indx, cell in dcells.iteritems():
		draw_cell(bmap, cell, color=color)
		#draw_cell_center(bmap, cell_bbx=cell, rank=indx)
	#plt.show()
	# plt.savefig('figs/road_active_cells_%skm.png' % lon_step, format='PNG', bbox_inches='tight')
	return plt


def number_to_color(v, oldMin, oldMax):
	"""
	Given a value v in terval [oldMin, oldMax] find it corresponding value nv in [0,120]
	Actually, we return 1 - nv.
	Objective: Get colors between red (0) and green (120), with yellow in the middle.
				The idea is to use Hue color system.
	:param v: value to be transformed, emission of CO2
	:param oldMin: the min emission value observed in the list of values
	:param oldMax: the max emission value observed in the list of values
	:return: a new value nv of v in the interval [0, 120]
	"""

	newMin = 0
	newMax = 120
	return 120 - ((((v - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin)

def draw_colorful_polygones(fname, att_name='rings'):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	bmap = init_doha_map(bbox=qatar_bbox, shapefile=None)

	polygons_shp = []
	cnt = 0
	with open(fname, 'r') as f:
		admins = json.load(f)['features']
		for admin in admins:
			polygon = [esri_to_gps(p[0], p[1]) for p in admin['geometry'][att_name][0]]
			polygon_shp = {'type': 'Polygon', 'coordinates': [polygon]}
			polygons_shp.append(polygon_shp)


	mp = MultiPolygon([shape(pol) for pol in polygons_shp])

	cm = plt.get_cmap('RdBu')
	num_colours = len(mp)


	minx, miny, maxx, maxy = mp.bounds
	w, h = maxx - minx, maxy - miny
	#ax.set_xlim(minx - 0.2 * w, maxx + 0.2 * w)
	#ax.set_ylim(miny - 0.2 * h, maxy + 0.2 * h)
	#ax.set_aspect(1)


	patches = []
	for idx, p in enumerate(mp):
		colour = cm(1. * idx / num_colours)
		patches.append(PolygonPatch(p, fc=colour, ec='#555555', lw=0.2, alpha=1., zorder=1))
	ax.add_collection(PatchCollection(patches, match_original=True))

	plt.show()

#draw_colorful_polygones(fname)

#

if __name__ == '__main__':
	from MapGrid import MapGrid
	doha_bbox = [25.21558, 25.387415, 51.44717, 51.630657]
	lon_step = 2
	lat_step = 2
	grid = MapGrid(bbox=doha_bbox, lon_step=lon_step, lat_step=lat_step)
	print grid.CELLS
	point = [51.48, 25.30]
	cell = grid.find_cell(point=point)
	print 'cell:', cell, grid.CELLS[cell]
	print 'Index:', grid.INDEX
	bmap = init_doha_map(bbox=doha_bbox, shapefile=None)
	draw_dict_cells(bmap=bmap, dcells=grid.CELLS, bbox=grid.BBOX, cell_activation_rank=True, color='gold')
	draw_point(bmap=bmap, point=point, color='red', size=200)
	plt.show()

