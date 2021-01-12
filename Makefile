autoformat:
	dev/linter.sh format

lint:
	dev/linter.sh

test:
	pytest tests/

docs-build:
	cd docs; make html

dev:
	pip install black==19.3b0 isort==4.3.21 flake8 sphinx-rtd-theme nbsphinx recommonmark

all: autoformat lint test