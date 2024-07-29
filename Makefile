IN_PROJECT?=true


.PHONY: init
init:
	rm -r ./.venv || true
	poetry config virtualenvs.in-project ${IN_PROJECT}
	poetry install


.PHONY: mypy
mypy:
	poetry run mypy src

.PHONY: format
format:
	poetry run python3 -m ruff check --select I --fix
	poetry run python3 -m ruff format

.PHONY: test
test:
	poetry run pytest tests -m "not e2e_test" --cov=src --cov-report term-missing --durations 5


.PHONY: test
e2e_test:
	poetry run pytest tests -m "e2e_test"


.PHONY: lint
lint:
	poetry run python3 -m ruff check --diff
	# $(MAKE) mypy

.PHONY: dev-install
dev-install: init
	poetry run python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


.PHONY: document
document:
	rm -rf docs/build || true
	rm -rf docs/source/reference/generated || true
	poetry run sphinx-build -M html docs/source docs/build
