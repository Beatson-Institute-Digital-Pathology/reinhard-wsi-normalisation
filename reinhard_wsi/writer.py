import numpy as np
import sys
import tifffile as tf
import cv2
from tqdm import tqdm

class OME_TIF_Writer():
	def __init__(self, output_pth):
		self.output_pth = output_pth

	def write_arr(self, slide_arr:np.ndarray, compression_arg:str="jpeg", levels:int=8, level_zero_uncompressed=True) -> None:
		"""Writes whole slide numpy array to pyramidal ome tif file.

		Args:
			slide_arr (np.array): The wholeslide array to write to file.
			compression (str, optional): If none the data is written uncompressed. Otherwise either 'LZMA' or 'ZSTD' or 'jpeg' (jpeg requires the imagecodecs package to be installed). Defaults to None.
			levels (int, optional): [description]. The number of pyramidal sub resolutions to include. Defaults to 8.
		"""

		if level_zero_uncompressed:
			compression = None
		else:
			compression = compression_arg

		if sys.version_info.minor < 7:
			options = {"tile": (256, 256), "compress": compression}
		else:
			options = {"tile": (256, 256), "compression": compression}
		
		with tf.TiffWriter(self.output_pth, bigtiff=True) as tif:
			tif.save(slide_arr, subifds=levels, **options)
			options.update({"subfiletype":1, "compression":compression_arg})
			for _ in tqdm(range(0, levels), desc="Writing subresolutions"):
				slide_arr = cv2.resize(slide_arr,(slide_arr.shape[1] // 2, slide_arr.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
				tif.save(slide_arr, **options)
