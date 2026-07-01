.PHONY: install run test docker-up docker-down fmt

install:
	pip install -r requirements.txt

run:
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

docker-up:
	docker compose up --build

docker-down:
	docker compose down
