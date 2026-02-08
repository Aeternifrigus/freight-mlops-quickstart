.PHONY: data etl lint clean

data:
	python data/generate_data.py --rows 20000 --out data/raw_shipments.csv

etl:
	python etl/spark_etl.py --input data/raw_shipments.csv --output data/features.csv

lint:
	ruff check .

clean:
	rm -f data/raw_shipments.csv data/features.csv
