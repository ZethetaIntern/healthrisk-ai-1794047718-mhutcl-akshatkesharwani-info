.PHONY: install test lint train sim figures report docker all

install:
	pip install -r requirements.txt

test:
	pytest --cov=data --cov=models --cov=financial --cov=simulation --cov=explainability --cov-report=term-missing tests

lint:
	ruff check . --select E9,F63,F7,F82

train:
	python run_models.py

sim:
	python run_sim.py

figures:
	python make_figures.py

report:
	python make_report.py

docker:
	docker compose up --build

all: install train sim figures report test
