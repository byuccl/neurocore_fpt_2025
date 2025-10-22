zip_gat2_model:
	7z a -v99m zip/models/gat2/gat2.7z gat2_pairnorm.pt

unzip_gat2_model:
	7z x zip/models/gat2/gat2.7z.001 -omodels

unpack_datasets:
	mkdir -p datasets
	tar -xzvf zip/dataset1.tar.gz -C datasets
	tar -xzvf zip/dataset2.tar.gz -C datasets

env:
	python3.12 -m venv .venv
	. .venv/bin/activate; pip install -r requirements.txt