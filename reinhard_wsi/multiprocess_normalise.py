from os import stat
from multiprocessing import sharedctypes
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from reinhard_wsi.utilites import generate_coordinates, non_white_mask_clipped
import numpy as np 

class ArrayDepthException(Exception):
	def __init__(self, message="Numpy array should have shape (n,m,3). This function expects a LAB array."):
		super().__init__(self.message)

class ArrayTypeExcpeiton(Exception):
	def __init__(self, message="Numpy array is not of type UInt8."):
		super().__init__(self.message)

SHARED_LAB = None

def apply_channel(channel_arr:np.ndarray, channel_no:int, src_meta:dict, tgt_meta:dict) -> np.ndarray:
	channel_arr = channel_arr.astype(np.float32)
	channel_arr = (((channel_arr-src_meta.get("means")[channel_no])/src_meta.get("stdevs")[channel_no])*tgt_meta.get("stdevs")[channel_no])+tgt_meta.get("means")[channel_no]
	channel_arr = np.clip(channel_arr, 0, 255).astype(np.uint8)
	return channel_arr

def normalise(args:list) -> None:
	patch_coords, src_meta, tgt_meta = args
	shared_in = np.ctypeslib.as_array(SHARED_LAB)

	lab_patch = shared_in[patch_coords.ystart:patch_coords.yend,patch_coords.xstart:patch_coords.xend,:]

	L = apply_channel(lab_patch[:,:,0], 0, src_meta, tgt_meta)
	A = apply_channel(lab_patch[:,:,1], 1, src_meta, tgt_meta)
	B = apply_channel(lab_patch[:,:,2], 2, src_meta, tgt_meta)
	lab_stacked = np.dstack((L,A,B))

	shared_in[patch_coords.ystart:patch_coords.yend,patch_coords.xstart:patch_coords.xend,:] = lab_stacked

class LABReinhardNormaliser():

	@staticmethod
	def normalise_channels(LAB:np.array, src_meta:dict, tgt_meta:dict) -> np.array:
		global SHARED_LAB
		SHARED_LAB = None

		if LAB.dtype != np.uint8:
			raise ArrayTypeExcpeiton()
		
		dimensions = LAB.shape
		patch_coordinate_list = generate_coordinates(dimensions)
		args = [(patch_coords, src_meta, tgt_meta) for patch_coords in patch_coordinate_list]

		input_temp = np.ctypeslib.as_ctypes(LAB)
		SHARED_LAB = sharedctypes.RawArray(input_temp._type_, input_temp)

		process_map(normalise, args, chunksize=1, desc="Normalising WSI.") # max_workers=4

		return np.ctypeslib.as_array(SHARED_LAB)
	
	@staticmethod
	def clean_arrays():
		global SHARED_LAB
		del SHARED_LAB
		SHARED_LAB = None
