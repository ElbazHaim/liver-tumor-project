DATA_ROOT := "data"

install:
	pip install -r requirements.txt

dev_install: install
	pip install ruff pylint

clean:
	rm -rf build*
	rm -rf src/*.egg-info*

build_package:
	pip install .
	make clean

train: build_package
	train \
	--yaml /mnt/data/haim_asaf/code/liver_tumor_segmentation/src/training/.configs/segformer.yaml


prepare_dataset: build_package
	prepare_dataset \
	--data_root_dir /mnt/data/haim_asaf/data \
	--arrow_output_dir /mnt/data/haim_asaf/data/20241003_processed/arrow

download:
	kaggle datasets download -p $(DATA_ROOT) andrewmvd/liver-tumor-segmentation-part-1
	kaggle datasets download -p $(DATA_ROOT) andrewmvd/liver-tumor-segmentation-part-2

unzip:
	unzip liver-tumor-segmentation.zip
	unzip liver-tumor-segmentation-part-2.zip
	mkdir volumes && mv volume_pt*/* volumes
	rm -rf volume_pt* && rm *.zip
