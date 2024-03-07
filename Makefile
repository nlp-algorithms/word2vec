.PHONY: install
BIN=.venv/bin/
PYTHON=python3

test:
	# python -m unittest discover tests/

install:
	$(PYTHON) -m venv .venv
	$(BIN)pip install -r requirements.txt
	$(BIN)pip install .

clean:
	rm -rf .venv
	find . -iname "*.pyc" -delete