from common import *
import sys
def find_sample(spid=None, rand_pt=None, neighbors=None, draw_output=False):
	if len(neighbors) < 2:
		return SamplePoint(spid=spid, lon=rand_pt.lon, lat=rand_pt.lat, speed=rand_pt.speed, angle=rand_pt.angle,
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
		if eq_perpend.slope == float('Inf'):
			Sx = eq_perpend.intercept
		else:
			Sx = (Sy - eq_perpend.intercept) / eq_perpend.slope
	if Sx < 30:
		print 'BUG HERE', axis_label, eq_perpend, Sx

	if draw_output == False:
		avg_speed = sum([nei.speed for nei in neighbors]) / len(neighbors)
		avg_heading = sum([nei.angle for nei in neighbors]) / len(neighbors)
		return SamplePoint(spid=spid, lon=Sx, lat=Sy, speed=avg_speed, angle=avg_heading, weight=len(neighbors))

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


# ---------------------------------------------------------------------------------------------
# ----------------------------TRAJECTORY MEAN-SHIFT SAMPLING-----------------------------------
# ---------------------------------------------------------------------------------------------

def traj_meanshift_sampling(filecode=None, RADIUS_METER=25, HEADING_ANGLE_TOLERANCE=2.5):
	"""
	Generate samples from raw GPS data.
	:param INPUT_FILE_NAME: tab separated file as generated from QMIC data
	:param RADIUS_METER: threshold distance between points and their center (sample) in meters.
	:param HEADING_ANGLE_TOLERANCE: threshold angle for points assumed to be heading in the same direction.
	:return: mapping point to sample!
	"""
	RADIUS_DEGREE = RADIUS_METER * 10e-6
	gpspoint_to_samples = dict()
	data_points, raw_points, points_tree = load_data(fname='data/gps_data/%s.csv' % filecode)
	print 'nb points:', len(data_points), 'points example:', data_points[0], raw_points[0]

	samples = list()
	removed_points = set()
	available_point_indexes = np.arange(0, len(data_points))
	spid_cnt = 1
	while len(available_point_indexes) > 0:
		sys.stdout.write('\rTHERE ARE %s points/ removed %s points ....' % (len(available_point_indexes), len(removed_points)))
		sys.stdout.flush()
		rand_index = random.sample(available_point_indexes, 1)[0]
		rand_pt = data_points[rand_index]
		# Find all neighbors in the given radius: RADIUS
		neighbor_indexes = [rand_index] + retrieve_neighbors(in_pt=rand_pt.get_coordinates(), points_tree=points_tree,
		                                                     radius=RADIUS_DEGREE)
		remaining_neighbor_indexes = list(set(neighbor_indexes) - removed_points)
		neighbor_indexes = []
		neighbors = []
		# find neighbors that are within a certain angle. This will eliminate points from the opposite direction.
		for val in remaining_neighbor_indexes:
			if abs(data_points[val].angle - rand_pt.angle) <= HEADING_ANGLE_TOLERANCE:
				neighbors.append(data_points[val])
				neighbor_indexes.append(val)
		# call find center method
		sample_pt = find_sample(spid=spid_cnt, rand_pt=rand_pt, neighbors=neighbors, draw_output=False)
		if sample_pt.lon < 50:
			print 'PROBLEM:', len(neighbor_indexes), rand_pt, sample_pt
		samples.append(sample_pt)
		for nei in neighbors:
			gpspoint_to_samples[nei.ptid] = spid_cnt
		spid_cnt += 1
		# Remove elements
		removed_points = removed_points.union(neighbor_indexes)
		available_point_indexes = sorted(set(available_point_indexes) - removed_points)


	print 'NB SAMPLES: %s' % len(samples)
	# store results into a file
	json.dump(to_geojson(samples), open('data/%s_samples.geojson' % (
		filecode), 'w'))
	json.dump(gpspoint_to_samples, open('data/%s_mappings_point_to_sample.json' % (
		filecode), 'w'))

	# json.dump(to_geojson(samples), open('data/%s_samples_r%s_a%s.geojson' % (
	# 	OUTPUT_FILE_CODE, RADIUS_METER, HEADING_ANGLE_TOLERANCE), 'w'))
	# json.dump(gpspoint_to_samples, open('data/%s_mappings_point_to_sample_r%s_a%s.json' % (
	# 	OUTPUT_FILE_CODE, RADIUS_METER, HEADING_ANGLE_TOLERANCE), 'w'))
	#return gpspoint_to_samples

	# # Plot results.
	# for s in samples:
	# 	if s.lon > 50:
	# 		plt.scatter(s.lon, s.lat)
	# # Get x/y-axis in the same aspect
	# plt.gca().set_aspect('equal', adjustable='box')
	# plt.show()


# ---------------------------------------------------------------------------------------------
# ----------------------------ROAD SEGMENT CLUSTERING------------------------------------------
# ---------------------------------------------------------------------------------------------

def road_segment_clustering(filecode=None, samples=None, minL=100, dist_threshold=50, angle_threshold=30):
	"""
	Create a graph from the samples. Nodes are samples, edges are road segments.
	:param samples:
	:param minL:
	:param dist_threshold_degree: distance threshold in meters
	:param angle_threshold:
	:return:
	"""
	dist_threshold_degree = dist_threshold * 10e-6
	if samples is None:
		data = json.load(open('data/%s_samples.geojson' % (filecode)))
		samples, sample_dict = to_samplepoints(data)

	print 'NB nodes:', len(samples)
	# 1. Sort samples by weight, and create KD-Tree
	samples.sort(key=operator.attrgetter('weight'), reverse=True)
	simple_samples = [(s.lon, s.lat) for s in samples]
	samples_tree = cKDTree(simple_samples)

	# 2. for each element Si find Sp and Sq to form a smooth segment Sp-->Si-->Sq
	# node_degree = {_:0 for _ in range(len(samples))} # dictionary of node: degree
	g = nx.DiGraph()
	for ind, s in enumerate(samples):
		g.add_node(ind, id=s.spid, speed=s.speed, angle=s.angle, lon=s.lon, lat=s.lat, weight=s.weight)

	for i, Si in enumerate(samples):
		print i, 'processing: ', Si
		if g.degree(i) >= 2:
			continue

		neighbors_index = [ind for ind in retrieve_neighbors(simple_samples[i], samples_tree, radius=dist_threshold_degree)\
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
			if g.degree(Sp_index) < 2:
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
			if g.degree(Sq_index) < 2:
				g.add_edge(i, Sq_index)


		# Account for cases where the sample point has no neighbor and belong to no segemets!
		# TODO: check later if this reflexive link doesn't cause other problems
		# if len(Sq_candidates_index) + len(Sp_candidates_index) == 0:
		# 	g.add_edge(i, i)


	#segments = sorted(nx.weakly_connected_components(g), key=len, reverse=True)
	paths = get_paths(g)
	print 'NB edges:', len(g.edges()), 'NB segments:', len(paths), 'NB nodes:', len(g)

	geojson = segments_to_geojson(paths, g)
	json.dump(geojson, open('data/%s_segments.geojson' % (filecode), 'w'))

	# build mapping between samples and segments.
	sample_to_segment = {g.node[p]['id']: segment_id for segment_id, path in enumerate(paths) for p in path}
	json.dump(sample_to_segment, open('data/%s_mappings_sample_to_segment.json' % (filecode), 'w'))

# ---------------------------------------------------------------------------------------------
# ----------------------------INFERRING LINKS BETWEEN SEGMENTS---------------------------------
# ---------------------------------------------------------------------------------------------

def inferring_links_between_segments(filecode=None, samples=None, segments=None, points_to_samples=None, samples_to_segments=None,
                                     max_link_length=100):
	"""
	I assume that this is simply adding edges to the graph!
	The idea is to take all trajectories, and process them one by one.
	For each trajectory, create the missing links.
	Inputs:
	1. gps points should have ids.
	2. sample points should have ids.
	3. segments should have ids.
	4. dict: gps_point to sample
	5. dict: sample to segment.
	:param max_link_length: the maximum length in meter of the link to be created.
	:return:
	"""
	# read/generate trajectories. each trajectory is: [pt1, pt2, ... ptn]
	trajectories = create_trajectories(INPUT_FILE_NAME='data/gps_data/%s.csv' % filecode, waiting_threshold=21)
	# read the mappings points to samples:
	if points_to_samples is None:
		points_to_samples = {int(k):v for k,v in json.load(open('data/%s_mappings_point_to_sample.json' % filecode)).iteritems()}
	# read the mappings samples to segments:
	if samples_to_segments is None:
		samples_to_segments = {int(k): v for k, v in json.load(open('data/%s_mappings_sample_to_segment.json' % filecode)).iteritems()}
	# read samples and segments. Each segment is a list of sample points: [s1, s2, ...sl]
	if samples is None:
		data = json.load(open('data/%s_samples.geojson' % filecode))
		samples, samples_dict = to_samplepoints(data)
	if segments is None:
		data = json.load(open('data/%s_segments.geojson' % filecode))
		segment_pt_coordinates, segments = to_segments(data)
	print 'samples_numbers:', len(set(points_to_samples.values())), len(samples_to_segments.keys())

	links = []
	for traj in trajectories:
		# the idea here is to assign each point to the closest sample point.
		# 1. map gps points in trajectories into samples.
		traj_samples = uniquify([points_to_samples[p.ptid] for p in traj])

		# 2. map samples into segments: in some cases some samples are missing from segments if they are not connected to other samples
		traj_segments = uniquify([samples_to_segments[s] for s in traj_samples if s in samples_to_segments.keys()])

		# 3. create an edge between every successive segments: last node in s_i with first element in s_i+1
		# TODO: I should create a link only if the distance between two nodes is lower than a given threshold distance.
		links += [(segments[traj_segments[i]][-1], segments[traj_segments[i + 1]][0]) for i in range(len(traj_segments) - 1)]

	# get geojson of new links with their weight computed as their frequencies.
	geojson = links_to_geojson(links, samples_dict, max_link_length)
	json.dump(geojson, open('data/%s_links.geojson' % filecode, 'w'))


def create_final_graph(segments_fname, links_fname):
	g = nx.DiGraph()
	data = json.load(open(segments_fname))
	segment_pt_coordinates, segments = to_segments(data)
	for id, seg in segment_pt_coordinates.iteritems():
		print len(seg[0]), seg[0]
		for i in range(len(seg[0]) - 1):
			g.add_edge(tuple(seg[0][i]), tuple(seg[0][i+1]))

	data = json.load(open(links_fname))
	links_pt_coordinates, links = to_links(data)
	for id, link in links_pt_coordinates.iteritems():
			g.add_edge(tuple(link[0][0]), tuple(link[0][1]))

	with open('data/kdd16_edges.txt', 'w') as fout:
		for s,t in g.edges():
			fout.write('%s,%s\n%s,%s\n\n' % (s[0], s[1], t[0], t[1]))



if __name__ == "__main__":

	# 0. Parameters:
	# --------------
	INPUT_FILE_NAME = 'data/gps_data/gps_points_07-11.csv'
	FILE_CODE = 'gps_points_07-11'
	RADIUS_METER = 25
	HEADING_ANGLE_TOLERANCE = 3

	# ---------------
	minL = 100
	dist_threshold = 200
	angle_threshold = 60

	# ---------------
	max_link_length = 200
	# -----------------
	# 1. find samples
	# traj_meanshift_sampling(filecode=FILE_CODE, RADIUS_METER=RADIUS_METER, HEADING_ANGLE_TOLERANCE=HEADING_ANGLE_TOLERANCE)

	# 2. find segments
	# road_segment_clustering(filecode=FILE_CODE, minL=100, dist_threshold=dist_threshold, angle_threshold=angle_threshold)

	# 3. Inferring links between segments
	inferring_links_between_segments(filecode=FILE_CODE, samples=None, segments=None, points_to_samples=None,
	                                 samples_to_segments=None, max_link_length=max_link_length)

	# 4. Write output to the file:
	create_final_graph(segments_fname='data/gps_points_07-11_segments.geojson', links_fname='data/gps_points_07-11_links.geojson')