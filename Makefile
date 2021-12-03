autoformat:
	dev/linter.sh format

lint:
	dev/linter.sh

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=./ --cov-report=xml

docs-build:
	cd docs; make html

build-dev:
	pip install --upgrade black==19.3b0 isort==4.3.21 flake8 flake8-bugbear flake8-comprehensions sphinx-rtd-theme nbsphinx recommonmark

all: autoformat lint test