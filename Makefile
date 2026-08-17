.PHONY: help install test lint gate view apply-view db-up db-down ingest train app docker-up docker-down clean

help:
	@echo "install      - pip install requirements"
	@echo "test         - run the full pytest suite (coverage-gated at 85%)"
	@echo "lint         - flake8 over src/ and tests/"
	@echo "gate         - lint + generated-SQL-drift check + tests (the CI gate, locally)"
	@echo "view         - regenerate db/metrics_view.sql from the feature registry"
	@echo "apply-view   - apply db/metrics_view.sql to the live DATABASE_URL (after first ingest)"
	@echo "db-up        - start a local Postgres via docker-compose"
	@echo "db-down      - stop it"
	@echo "ingest       - run scripts/ingest_data.py against DATABASE_URL"
	@echo "train        - run scripts/train_model.py against DATABASE_URL"
	@echo "app          - run the Streamlit app locally"
	@echo "docker-up    - full stack: Postgres + Streamlit via docker-compose"
	@echo "docker-down  - stop the full stack"
	@echo "clean        - remove caches and coverage artifacts"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	flake8 src tests scripts/generate_metrics_view.py

view:
	python scripts/generate_metrics_view.py

view-check:
	python scripts/generate_metrics_view.py --check

apply-view:
	python scripts/apply_metrics_view.py

gate: lint view-check test

db-up:
	docker-compose up -d postgres

db-down:
	docker-compose stop postgres

ingest:
	python scripts/ingest_data.py

train:
	python scripts/train_model.py

app:
	streamlit run src/f1_pit_window/app/streamlit_app.py

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov
