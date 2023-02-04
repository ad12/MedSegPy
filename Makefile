autoformat:
	set -e
	isort .
	black --config pyproject.toml .
	flake8

lint:
	set -e
	isort -c .
	black --check --config pyproject.toml .
	flake8

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=./ --cov-report=xml

docs-build:
	cd docs; make html

build-dev:
	pip install --upgrade black==22.3.0 isort flake8 flake8-bugbear flake8-comprehensions sphinx-rtd-theme nbsphinx recommonmark pooch coverage

all: autoformat lint test