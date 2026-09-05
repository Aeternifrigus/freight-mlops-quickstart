.PHONY: data etl train test-lambda all clean

data:
	python data/generate_data.py --rows 20000 --out data/raw_shipments.csv

etl:
	python etl/spark_etl.py --input data/raw_shipments.csv --output data/features.csv

train:
	python train/train_model.py --input data/features.csv --out train/model_bundle.joblib
	cp train/model_bundle.joblib serve/model_bundle.joblib
	python etl/build_carrier_reference.py --input data/features.csv --out serve/carrier_reference.json

test-lambda:
	cd serve && python test_local.py

all: data etl train test-lambda

clean:
	rm -f data/raw_shipments.csv data/features.csv train/model_bundle.joblib \
	      train/metrics.json serve/model_bundle.joblib
