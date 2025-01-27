IN_PROJECT?=true
VERSION=`poetry version --short`

.PHONY: reset
reset:
	rm -r ./.venv || true
	rm poetry.lock || true
	poetry config virtualenvs.in-project ${IN_PROJECT}


.PHONY: install_cu124
install_cu124:
	poetry sync --no-root -E cu124 --with cu124 


.PHONY: install_cpu
install_cpu:
	poetry sync --no-root -E cpu --with cpu


.PHONY: mypy
mypy:
	poetry run mypy src

.PHONY: format
format:
	poetry run python3 -m ruff format
	poetry run python3 -m ruff check --fix

.PHONY: test
test:
	poetry run pytest tests -m "not e2e_test and not gpu_test" --cov=src --cov-report term-missing --durations 5


.PHONY: e2e_test
e2e_test:
	poetry run pytest tests -m "e2e_test"


.PHONY: gpu_test
gpu_test:
	poetry run pytest tests -m "gpu_test"


.PHONY: lint
lint:
	poetry run python3 -m ruff check --output-format=full
	poetry run python3 -m ruff format --diff
	# $(MAKE) mypy


.PHONY: document
document:
	rm -rf docs/build || true
	rm -rf docs/source/reference/generated || true
	rm -rf docs/source/tutorials/basic_usages || true
	poetry run sphinx-build -M html docs/source docs/build


.PHONY: push_docker_images
push_docker_images: 
	make -C docker push VERSION=${VERSION}