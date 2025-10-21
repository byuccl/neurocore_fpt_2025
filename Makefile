zip_gat2_model:
	7z a -v99m zip/models/gat2/gat2.7z gat2_pairnorm.pt

unzip_gat2_model:
	7z x zip/models/gat2/gat2.7z.001 -C models

unpack_datasets:
	mkdir -p datasets
	tar -xzvf zip/dataset1_tcl_25dumps.tar.gz -C datasets
	tar -xzvf zip/dataset2_tcl_25dumps.tar.gz -C datasets
