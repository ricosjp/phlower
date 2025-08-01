IN_PROJECT?=true
VERSION=`poetry version --short`

.PHONY: reset
reset:
	rm -r ./.venv || true
	rm poetry.lock || true
	poetry config virtualenvs.in-project ${IN_PROJECT}


.PHONY: install
install:
	poetry sync --no-root


.PHONY: mypy
mypy:
	poetry run mypy src

.PHONY: format
format:
	poetry run ruff format
	poetry run ruff check --fix

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
	poetry run ruff check --output-format=full
	poetry run ruff format --diff
	# $(MAKE) mypy


.PHONY: examples
examples:
	$(foreach file, \
		$(wildcard examples/**/main.py), \
		cd $(shell dirname $(file)); \
		poetry run python3 main.py; \
		cd -; \
		)


.PHONY: document
document:
	rm -rf docs/build || true
	rm -rf docs/source/reference/generated || true
	rm -rf docs/source/tutorials/basic_usages || true
	poetry run sphinx-build -M html docs/source docs/build


.PHONY: push_docker_images
push_docker_images: 
	make -C docker push VERSION=${VERSION}
