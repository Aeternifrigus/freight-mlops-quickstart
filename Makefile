.PHONY: data lint clean

data:
	python data/generate_data.py --rows 20000 --out data/raw_shipments.csv

lint:
	ruff check .

clean:
	rm -f data/raw_shipments.csv
