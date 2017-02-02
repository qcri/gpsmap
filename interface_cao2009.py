# author: Sofiane
# date: 2017-01-17
#
# This is an interface with cao2009 code.
# create files in a format that cao script car digest. One file per trip.
#

from common import create_trajectories
import time
import json
import os
import networkx as nx
import sys


def prepare_trips(INPUT_FILE_NAME='data/gps_data/gps_points_07-11.csv', waiting_threshold=600):
	trips = create_trajectories(INPUT_FILE_NAME=INPUT_FILE_NAME, waiting_threshold=waiting_threshold)
	print 'There are:', len(trips), ' trips\n'
	cnt = 0
	for trip in trips:
		if len(trip) < 2:
			continue
		print 'trip:', cnt, len(trip)
		with open('data/trips/trip_%s.txt' % cnt, 'w') as g:
		#with open('data/trips_%s/trip_%s.txt' % (INPUT_FILE_NAME.split('/')[-1], cnt), 'w') as g:
			for i, loc in enumerate(trip):
				if i == 0:
					prev_loc = 'None'

				if i == len(trip) - 1:
					next_loc = 'None'
				else:
					next_loc = trip[i + 1].locid

				g.write('%s,%s,%s,%s,%s,%s\n' % (loc.locid, loc.lon, loc.lat, time.mktime(loc.timestamp.timetuple()), prev_loc, next_loc))
				prev_loc = loc.locid
		cnt += 1

def check_bidirectionality(edge_fname='map_inference_algorithms/cao_edges.txt'):
	with open(edge_fname, 'r') as f:
		lines = [line for line in f]

	# build edges
	edges = []
	for i in range(len(lines) / 3):
		edge_lines = lines[3 * i: 3 * (i + 1)]
		source = '_'.join([_.strip() for _ in edge_lines[0].strip().split(',')])
		target = '_'.join([_.strip() for _ in edge_lines[1].strip().split(',')[:2]])
		edges.append((source, target))

	# check for indirection
	nb_reflex_edges = 0
	for s,t in edges:
		if (t,s) in edges:
			nb_reflex_edges += 1
	print nb_reflex_edges


def save_edges_to_geojson(cao_graph_edges_file='cao_edges.txt'):
	path = 'map_inference_algorithms'
	with open('%s/%s' %(path, cao_graph_edges_file), 'r') as f:
		lines = [line for line in f]


	# geojson = {'type': 'FeatureCollection', 'features': []}
	# for segment_id, s in enumerate(segments):
	# 	feature = {'type': 'Feature', 'properties': {}, 'geometry': {'type': 'LineString', 'coordinates': []}}
	# 	coordinates = [[g.node[p]['lon'], g.node[p]['lat']] for p in list(s)]
	# 	pt_ids = [g.node[p]['id'] for p in list(s)]
	# 	feature['geometry']['coordinates'] = coordinates
	# 	feature['properties'] = {'segment_id': segment_id, 'pt_ids': pt_ids}
	# 	geojson['features'].append(feature)

	geojson = {'type': 'FeatureCollection', 'features': []}
	for i in range(len(lines)/3) :
		edge_lines = lines[3 * i: 3 * (i + 1)]
		feature = {'type': 'Feature', 'properties': {}, 'geometry': {'type': 'LineString', 'coordinates': []}}
		coordinates = [[float(_) for _ in edge_lines[0].strip().split(',')], [float(_) for _ in edge_lines[1].strip().split(',')[:2]]]
		feature['geometry']['coordinates'] = coordinates
		geojson['features'].append(feature)
	json.dump(geojson, open('%s/%s' % (path, cao_graph_edges_file.replace('.txt', '.geojson')), 'w'))


def save_trips_points_to_geojson(trips_path='data/trips', prefix='', max_trips=200):
	path = 'map_inference_algorithms/map_inference_algorithms'
	all_trip_filenames = os.listdir(trips_path)
	relevant_trip_filenames = ['trip_%s.txt' % i for i in range(max_trips)]
	trip_filenames = list(set(all_trip_filenames).intersection(relevant_trip_filenames))

	geojson = {'type': 'FeatureCollection', 'features': []}

	for trip_file in trip_filenames:
		with open('%s/%s' % (trips_path, trip_file), 'r') as f:
			for line in f:
				locid, lon, lat, timestamp, prev_loc, next_loc = line.strip().split(',')
				feature = {'type': 'Feature', 'properties': {}, 'geometry': {'type': 'Point', 'coordinates': []}}
				feature['geometry']['coordinates'] = [float(lon), float(lat)]
				feature['properties']['id'] = locid
				geojson['features'].append(feature)
	json.dump(geojson, open('%s/%s_%s' % (path, prefix, 'trip_points.geojson'), 'w'))

if __name__ == '__main__':
	import sys

	fname = sys.argv[1]

	prepare_trips(INPUT_FILE_NAME='data/%s' % fname, waiting_threshold=21)

	# save_edges_to_geojson(cao_graph_edges_file='cao_edges.txt')
	# check_bidirectionality(edge_fname='map_inference_algorithms/cao_edges.txt')
	# sys.exit()

	nb_trips = 1000
	round_n = 0
	#save_trips_points_to_geojson(trips_path='data/trips', prefix='original', max_trips=nb_trips)
	#save_trips_points_to_geojson(trips_path='map_inference_algorithms/map_inference_algorithms/clarified_trips/n%s/round%s' % (nb_trips, round_n), prefix='clarified', max_trips=nb_trips)
