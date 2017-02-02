from common import *



def create_trajectories1(INPUT_FILE_NAME='data/gps_data/gps_points_07-11.csv', waiting_threshold=5):
	"""
	return all trajectories.
	The heuristic is simple. Consider each users sorted traces not broken by more than 1 hour as trajectories.
	:param waiting_threshold: threshold for trajectory split expressed in seconds.
	:return: list of lists of trajectories
	"""

	data_points, raw_points, points_tree = load_data(fname=INPUT_FILE_NAME)
	# data_points = load_data_rade(fname=INPUT_FILE_NAME)
	detections = defaultdict(list)
	for p in data_points:
		detections[p.btid].append(p)

	# compute trajectories: split detections by waiting_threshold

	print 'Computing trajectories for this number of cars:', len(detections)
	s = detections[2705322][0].timestamp
	nb_breaks = 0
	for p in sorted(detections[2705322], key=operator.attrgetter('timestamp')):
		d = p.timestamp - s
		d = d.days * 24 * 3600 + d.seconds
		if d > waiting_threshold:
			nb_breaks += 1
		s = p.timestamp
		print datetime.datetime.strftime(p.timestamp, '%Y-%m-%d %H:%M:%S'), d
	print 'Number of breaks:', nb_breaks

	trajectories = []
	total = 0

	btd = 2705322
	ldetections = detections[btd]
	btd_traj = 0
	points = sorted(ldetections, key=operator.attrgetter('timestamp'))
	source = 0
	prev_point = 0
	i = 1
	while i < len(points):
		delta = points[i].timestamp - points[prev_point].timestamp
		if delta.days * 24 * 3600 + delta.seconds > waiting_threshold:
			print 'Breaking trip: ', source, i
			trajectories.append(points[source: i])
			source = i
			btd_traj += 1
		prev_point = i
		i += 1

	if source < len(points):
		trajectories.append(points[source: -1])
		btd_traj += 1
	total += btd_traj

	print 'car: ', btd, ' has: ', btd_traj
	print 'total:', total
	return trajectories

trajectories = create_trajectories(INPUT_FILE_NAME='data/gps_data/gps_points_07-11.csv', waiting_threshold=21)
print len(trajectories)


