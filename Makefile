.PHONY: install lint test serve docker-build docker-run

install:
	uv sync --all-groups --extra serve

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest -q

serve:
	uv run uvicorn convgru_ensemble.serve:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t convgru-ensemble .

docker-run:
	docker run -p 8000:8000 -v ./checkpoints:/app/checkpoints convgru-ensemble
