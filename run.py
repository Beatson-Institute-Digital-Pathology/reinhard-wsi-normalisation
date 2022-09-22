from reinhard_wsi.normaliser import ReinhardNormaliser
import glob
import os


if __name__ == "__main__":

	input_dir = "/mnt/high_volume_storage/raw_datasets/SSL_CELL"

	file_list = glob.glob("{}/*.ndpi".format(input_dir))
	file_list.sort()
	
	output_dir = os.path.join(input_dir, "reinhard_normalised_v1-2")

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	for idx, slide_pth in enumerate(file_list):
		print("\n\n ------------")
		print("Slide No: {}/{}".format(idx+1, len(file_list)))
		basename = os.path.basename(slide_pth).replace(".ndpi", "")

		normalised_pth = os.path.join(output_dir, "{}.ome.tiff".format(basename))

		if os.path.exists(normalised_pth):
			print("Slide already normalised - Continuing: {}".format(slide_pth))
			continue

		normaliser = ReinhardNormaliser()
		
		# target = "/mnt/high_volume_storage/raw_datasets/INCISE/INC0294-2020-08-2816.55.44.ndpi"
		# normaliser.fit_target(target, level=1)
		# normaliser.save_target_stain_metadata(json_file)
		normaliser.load_fit("idealised_he_target_stain_metadata.json")

		normaliser.normalise(slide_pth, normalised_pth)