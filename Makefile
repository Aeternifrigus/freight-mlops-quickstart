.PHONY: lint clean

lint:
	ruff check .

clean:
	rm -f data/raw_shipments.csv
