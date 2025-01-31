CI_REGISTRY_IMAGE ?= registry.ritc.jp/ricos/machine_learning/phlower
VERSION ?= 


login:
ifeq ($(CI_JOB_TOKEN),)
	docker login $(CI_REGISTRY_IMAGE)
else
	docker login -u gitlab-ci-token -p $(CI_JOB_TOKEN) $(CI_REGISTRY_IMAGE)
endif


.PHONY: prepare
prepare:
	cp -f ../README.md dist
	cp -f ../pyproject.toml dist
	cp -f ../Makefile dist


.PHONY: push_cpu_images
push_cpu_images: prepare
	docker build \
		-t $(CI_REGISTRY_IMAGE):$(VERSION)-py310-cpu \
		--build-arg USE_PYTHON_VERSION=3.10 \
		-f cpu.Dockerfile \
		.
	docker push $(CI_REGISTRY_IMAGE):$(VERSION)-py310-cpu

	docker build \
		-t $(CI_REGISTRY_IMAGE):$(VERSION)-py311-cpu \
		--build-arg USE_PYTHON_VERSION=3.11 \
		-f cpu.Dockerfile \
		.
	docker push $(CI_REGISTRY_IMAGE):$(VERSION)-py311-cpu

	docker build \
		-t $(CI_REGISTRY_IMAGE):$(VERSION)-py312-cpu \
		--build-arg USE_PYTHON_VERSION=3.12 \
		-f cpu.Dockerfile \
		.
	docker push $(CI_REGISTRY_IMAGE):$(VERSION)-py312-cpu


.PHONY: build_gpu_image
build_gpu_image: prepare
	docker build \
		-t $(CI_REGISTRY_IMAGE):$(VERSION)-py311-cu124 \
		-f gpu.Dockerfile \
		.

.PHONY: push_gpu_image
push_gpu_image: build_gpu_image
	docker push $(CI_REGISTRY_IMAGE):$(VERSION)-py311-cu124


push: login push_cpu_images push_gpu_image
