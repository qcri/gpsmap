from MapGrid import MapGrid
from common import create_trajectories, assign_points_to_cells_v1, read_points
from MapPlotting import init_doha_map, draw_dict_cells
import pickle

# 1. Read checkins
shapefile = '../citymap/cityGrowth/data/doha_qatar.osm/doha_qatar_osm_polygon'
dates = range(11, 19)
dt = dates[0]
lon_step = 0.002
lat_step = 0.002
max_lon = 51.5024253
min_lon = 51.442650
max_lat = 25.329715
min_lat = 25.2655007
print '1. Create the grid'
doha_bbox = [min_lat, max_lat, min_lon, max_lon]
mgrid = MapGrid(bbox=doha_bbox, lon_step=lon_step, lat_step=lat_step)
#pickle.dump(mgrid, open('data/grid_%sx%s.pickle' % (lon_step, lat_step), 'w'))
# 2. Create cells
print '2. Cells created: %s' % (len(mgrid.CELLS))

for dt in dates:
	input_file = 'data/gps_data/gps_points_07-%s.csv' % dt
	# input_file = 'data/gps_data/3631800-gps_points_07-%s.csv' % dt
	# trajectories = create_trajectories(INPUT_FILE_NAME=input_file, waiting_threshold=5)
	points = read_points(fname=input_file)
	print '3. Read points 07-%s' % dt, ' Total: ', len(points)
	# min_lat = 100
	# max_lat = 1
	# min_lon = 100
	# max_lon = 1
	#
	# determine boundaries of the bbox
	# pts = {}
	# cpt = 0
	# for k, v in points.iteritems():
	# 	pts[k] = v
	# 	cpt += 1
	# 	if cpt > 1000:
	# 		break
	# points = pts
	# for k, p in points.iteritems():
	# 	lon, lat = p['geo'] # tuple (lon, lat)
	# 	if lon > max_lon:
	# 		max_lon = lon
	# 	elif lon < min_lon:
	# 		min_lon = lon
	#
	# 	if lat > max_lat:
	# 		max_lat = lat
	# 	elif lat < min_lat:
	# 		min_lat = lat

	# 3. Assign points to cells:
	cell_to_points, point_to_cell = assign_points_to_cells_v1(mgrid, points)
	print '4. %s point mapped to %s cells' % (len(points), len(cell_to_points))

	#4. Plotting the grid
	#bmap = init_doha_map(bbox=doha_bbox, shapefile='doha')
	active_cells = {cell: mgrid.CELLS[cell] for cell in cell_to_points.keys()}
	pickle.dump(active_cells, open('data/active_cells_07-%s.json' % dt, 'w'))
	bmap = init_doha_map(bbox=doha_bbox, shapefile=shapefile)
	plt = draw_dict_cells(bmap=bmap, dcells=active_cells, bbox=mgrid.BBOX, cell_activation_rank=False, color='blue', lon_step=lon_step)
	plt.savefig('figs/road_active_cells_07-%s.pdf' % dt, format='PDF', bbox_inches='tight')

# Visualization
# bmap = init_doha_map(bbox=doha_bbox, shapefile=shapefile)
# ac1 = pickle.load(open('data/active_cells_07-%s.json' % 11))
# ac2 = pickle.load(open('data/active_cells_07-%s.json' % 12))
#
# closed = set(ac1.keys()) - set(ac2.keys())
# newop = set(ac2.keys()) - set(ac1.keys())
# same = set(ac1.keys()).intersection(set(ac2.keys()))
# plt = draw_dict_cells(bmap=bmap, dcells={cell: mgrid.CELLS[cell] for cell in same}, bbox=mgrid.BBOX, cell_activation_rank=False, color='blue', lon_step=lon_step)
# plt = draw_dict_cells(bmap=bmap, dcells={cell: mgrid.CELLS[cell] for cell in newop}, bbox=mgrid.BBOX, cell_activation_rank=False, color='red', lon_step=lon_step)
# plt = draw_dict_cells(bmap=bmap, dcells={cell: mgrid.CELLS[cell] for cell in closed}, bbox=mgrid.BBOX, cell_activation_rank=False, color='black', lon_step=lon_step)
# plt.savefig('figs/activation_same_new1.pdf', format='PDF', bbox_inches='tight')
# plt.show()

