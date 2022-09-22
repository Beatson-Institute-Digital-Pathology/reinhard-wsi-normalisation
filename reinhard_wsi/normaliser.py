import tifffile as tf
import os
import cv2
import numpy as np
import random
import json

from reinhard_wsi.logging import get_logger
from reinhard_wsi.utilites import non_white_mask_clipped
from reinhard_wsi.multiprocess_normalise import LABReinhardNormaliser
from reinhard_wsi.writer import OME_TIF_Writer

from sklearn.cluster import KMeans

logger = get_logger("ReinhardNormaliser", "stain_normalisation.log")

class ReinhardNormaliser():
	def __init__(self) -> None:
		self.src_meta = None
		self.tgt_meta = None
		self.src_slide = None
		self.tgt_slide = None
		self.tissue_predictor = None

	def load_slide(self, slide_pth:str, level=0) -> np.ndarray:
		logger.debug("Loading slide: {}".format(slide_pth))
		slide_arr = tf.imread(slide_pth, series=0, level=level)
		logger.debug("Slide Shape: {}".format(slide_arr.shape))

		slide_arr = cv2.cvtColor(slide_arr, cv2.COLOR_RGBA2RGB)
		logger.debug("Slide Loaded.")
		return slide_arr

	def sample_slide(self, slide_arr:str, samples=1000000) -> list:
		slide_shape = slide_arr.shape
		y_coords = random.choices(range(slide_shape[0]), k=samples)
		x_coords = random.choices(range(slide_shape[1]), k=samples)
		sample_coords = zip(y_coords, x_coords)

		sampled_pixels = np.zeros((samples,3), dtype=np.uint8)

		for idx, (y,x) in enumerate(sample_coords):
			sampled_pixels[idx] = slide_arr[y,x,:]
		return sampled_pixels

	def get_channel_stats(self, lab_arr:np.ndarray, tissue_mask:np.ndarray, pixel_class:int=1, statistic=np.mean) -> tuple:
		L = lab_arr[:,:,0]
		A = lab_arr[:,:,1]
		B = lab_arr[:,:,2]

		L = L[tissue_mask==pixel_class]
		A = A[tissue_mask==pixel_class]
		B = B[tissue_mask==pixel_class]

		L_mean = statistic(L)
		A_mean = statistic(A)
		B_mean = statistic(B)

		return L_mean, A_mean, B_mean

	def _train_tissue_predictor(self, sample_arr:np.ndarray) -> np.ndarray:
		logger.debug("Training tissue predictor.")
		LAB = np.squeeze(sample_arr)
		
		self.tissue_predictor = KMeans(n_clusters=2).fit(LAB)
		cluster_centers = self.tissue_predictor.cluster_centers_

		flip_clusters = False
		if cluster_centers[0][0] < cluster_centers[1][0]:
			flip_clusters = True

		mask = self.tissue_predictor.predict(LAB)
		if flip_clusters:
			mask = np.absolute(mask-1)

		return np.expand_dims(mask, axis=0)

	def _fit(self, slide_pth:str, target=False, level:int=0) -> dict:
		slide_arr = self.load_slide(slide_pth, level)
		sampled_pixels = self.sample_slide(slide_arr)

		lab_arr = cv2.cvtColor(np.expand_dims(sampled_pixels, axis=0), cv2.COLOR_RGB2Lab)

		tissue_mask = self._train_tissue_predictor(lab_arr)

		if target:
			percentile = np.percentile(lab_arr[:, :, 0], 98)
			normed_l = lab_arr[:, :, 0].astype(np.float32)/percentile
			L = np.clip(normed_l*255, 0, 255).astype(np.uint8)
			lab_arr[:, :, 0] = L
		
		means = self.get_channel_stats(lab_arr, tissue_mask, 1)
		stdevs = self.get_channel_stats(lab_arr, tissue_mask, 1, np.std)

		return {"means":means, "stdevs":stdevs}, slide_arr

	def fit_source(self, slide_pth:str, level:int=0) -> dict:
		self.src_meta, slide_arr = self._fit(slide_pth, level=level)
		return self.src_meta, slide_arr

	def fit_target(self, slide_pth:str, level:int=0) -> dict:
		self.tgt_meta, slide_arr = self._fit(slide_pth, target=True, level=level)
		return self.tgt_meta, slide_arr
	
	def write_fit(self, path:str) -> None:
		with open(path, "w") as jfile:
			json.dump(self.tgt_meta, jfile)
	
	def load_fit(self, path:str) -> None:
		with open(path, "r") as jfile:
			self.tgt_meta = json.load(jfile)

	def _write_ome_tif(self, array:np.ndarray, output_pth:str, xres:float, yres:float) -> None:
		logger.debug("Writing translated slide to .ome.tiff file.")
		writer = OME_TIF_Writer(output_pth)
		writer.write_arr(array, xres, yres, level_zero_uncompressed=True)

	def normalise(self, slide_pth:str, output_pth:str) -> None:
		src_meta, slide_arr = self.fit_source(slide_pth)

		slide_arr = cv2.cvtColor(slide_arr, cv2.COLOR_RGB2Lab)
		logger.debug("WSI RGB -> LAB")

		normalised_slide_arr = LABReinhardNormaliser.normalise_channels(slide_arr, self.src_meta, self.tgt_meta)
		LABReinhardNormaliser.clean_arrays()
		logger.debug("WSI Normalised.")

		slide_arr = cv2.cvtColor(normalised_slide_arr, cv2.COLOR_Lab2RGB)
		logger.debug("WSI LAB -> RGB")

		self._write_ome_tif(slide_arr, output_pth)
