.PHONY: all test
all:
	python pipeline.py

test:
	python -m unittest discover -s tests -v
