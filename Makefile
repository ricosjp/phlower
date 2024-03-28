IN_PROJECT?=true


.PHONY: init
init:
	rm -r ./.venv || true
	poetry config virtualenvs.in-project ${IN_PROJECT}
	poetry install


.PHONY: black-check
black-check:
	poetry run black --check src tests

.PHONY: black
black:
	poetry run black src tests

.PHONY: flake8
flake8:
	poetry run flake8 src tests

.PHONY: isort-check
isort-check:
	poetry run isort --check-only src tests

.PHONY: isort
isort:
	poetry run isort src tests

.PHONY: mypy
mypy:
	poetry run mypy src

.PHONY: format
format:
	$(MAKE) black
	$(MAKE) isort

.PHONY: test
test:
	poetry run pytest tests --cov=src --cov-report term-missing --durations 5

.PHONY: lint
lint:
	$(MAKE) black-check
	$(MAKE) isort-check
	$(MAKE) flake8
	$(MAKE) mypy

.PHONY: dev-install
dev-install: init
	poetry run python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
