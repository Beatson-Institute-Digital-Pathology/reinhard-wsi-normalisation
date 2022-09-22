from operator import mod
import cv2
import numpy as np
from collections import namedtuple

PatchCoordinate = namedtuple("PatchCoordinate", ["xstart", "ystart", "xend", "yend"])

def non_white_mask_clipped(lab_arr, upper_thresh=0.8, percentile=95, p_val=None):
	L = lab_arr[:, :, 0]
	if p_val is None:
		p_val = np.percentile(L, percentile)
	L = L.astype(np.float32)
	L = L/p_val
	mask =  L < upper_thresh
	return mask, p_val

def generate_coordinates(dimensions:tuple, patch_size:tuple=(5000,5000), xstart:int=0, ystart:int=0) -> list:
	patch_coordinates = []

	ymax = dimensions[0]
	xmax = dimensions[1]
	patchy = patch_size[1]
	patchx = patch_size[0]

	ydiff = ymax % patch_size[1]
	xdiff = xmax % patch_size[0]

	ymain_end = (ymax - ydiff)
	xmain_end = (xmax - xdiff)

	for y in range(ystart, ymain_end, patchy):
		for x in range(xstart, xmain_end, patchx):
			coordinate = PatchCoordinate(x, y, x+patchx, y+patchy)
			patch_coordinates.append(coordinate)

	if ydiff != 0:
		for x in range(xstart, xmain_end, patchx):
			coordinate = PatchCoordinate(x, ymain_end, x+patchx, ymax)
			patch_coordinates.append(coordinate)

	if xdiff != 0:
		for y in range(ystart, ymain_end, patchy):
			coordinate = PatchCoordinate(xmain_end, y, xmax, y+patchy)
			patch_coordinates.append(coordinate)

	if ydiff != 0 and xdiff != 0:
		coordinate = PatchCoordinate(xmain_end, ymain_end, xmax, ymax)
		patch_coordinates.append(coordinate)
				
	return patch_coordinates