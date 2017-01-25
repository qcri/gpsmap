""" 1. create grid of Doha.
 2. foreach day:
    - assign points to small cells 1x1m
    - plot grid.
"""
from MapGrid import MapGrid
from MapPlotting import init_doha_map, draw_cell, number_to_color, draw_cell_center

	# doha_bbox = [25.21558, 25.387415, 51.44717, 51.630657]


	lon_step = 1
	lat_step = 1
	grid = MapGrid(bbox=qatar_bbox, lon_step=lon_step, lat_step=lat_step)