from utils.fileGen import fileGen



FEATURE_PATH = "raw_data\\feature_matrix.csv"
OUTPUT_PATH = "raw_data\\output_matrix.csv"
IMAGE_PATH = "raw_data\\mapbox_api\\"
tofile = True
if tofile:
    file_generator = fileGen(FEATURE_PATH, OUTPUT_PATH)
    file_generator.generate_files()