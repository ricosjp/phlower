IN_PROJECT?=true
VERSION=`uv version --short`
IMAGE_VERSION = 0.4.0
CUDA_TAG = cu124


.PHONY: reset
reset:
	rm -r ./.venv || true
	rm uv.lock || true

.PHONY: install
install:
	uv sync --extra ${CUDA_TAG} --group dev


.PHONY: mypy
mypy:
	uv run mypy src

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix


.PHONY: test
test:
	uv run pytest tests -m "not e2e_test" --device=cpu --cov=src --cov-report term-missing --durations 5

.PHONY: dev_test_lf
test_lf:
	uv run pytest tests -m "not e2e_test" --lf


.PHONY: e2e_test
e2e_test:
	uv run pytest tests -m "e2e_test"


.PHONY: gpu_test
gpu_test:
	uv run pytest tests -m "gpu_test and not e2e_test" --device="cuda:0"


.PHONY: lint
lint:
	uv run ruff check --output-format=full
	uv run ruff format --diff
	# $(MAKE) mypy


.PHONY: examples
examples:
	$(foreach file, \
		$(wildcard examples/**/main.py), \
		cd $(shell dirname $(file)); \
		uv run python3 main.py || exit 1; \
		cd -; \
		)


.PHONY: document
document:
	rm -rf docs/build || true
	rm -rf docs/source/reference/generated || true
	rm -rf docs/source/tutorials/basic_usages || true
	uv run sphinx-build -M html docs/source docs/build


.PHONY: push_docker_images
push_docker_images: 
	make -f Makefile.docker push
