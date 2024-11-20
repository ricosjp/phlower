CI_REGISTRY_IMAGE ?= registry.ritc.jp/ricos/machine_learning/phlower
PROJECT_VERSION ?= 0.0.1

ifdef CI_COMMIT_TAG
	LOCAL_TAG = $(CI_COMMIT_TAG)
  LOCAL_TAG := $(shell echo $(TAG) | sed -e 's/\//__/')
else
  ifdef CI_COMMIT_REF_SLUG
	  LOCAL_TAG ?= $(CI_COMMIT_REF_SLUG)
  else
	  LOCAL_TAG := latest
  endif
endif

TAG := $(LOCAL_TAG):$(PROJECT_VERSION)

login:
ifeq ($(CI_JOB_TOKEN),)
	docker login $(CI_REGISTRY_IMAGE)
else
	docker login -u gitlab-ci-token -p $(CI_JOB_TOKEN) $(CI_REGISTRY_IMAGE)
endif


push: login
	docker build -t $(CI_REGISTRY_IMAGE)/$(TAG) -f Dockerfile .
	docker push $(CI_REGISTRY_IMAGE)/$(TAG)
