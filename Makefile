SHELL:=/bin/bash

# ---
# Variables
# ---
# include .env
# export $(shell sed 's/=.*//' .env)

PROJECT_PATH := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
GIT_REMOTE=$(shell basename $(shell git remote get-url origin))
PROJECT_NAME=$(shell echo $(GIT_REMOTE:.git=))
CURRENT_VERSION=$(shell git tag -l --sort=-creatordate | head -n 1 | cut -d "v" -f2-)

DOCKER_IMAGE_NAME=hsteinshiromoto/${PROJECT_NAME}

BUILD_DATE = $(shell date +%Y%m%d-%H:%M:%S)

ifndef DOCKER_TAG
DOCKER_TAG=${CURRENT_VERSION}
endif

ifndef DOCKER_PARENT_IMAGE
DOCKER_PARENT_IMAGE="python:3.9-slim"
endif

# ---
# Commands
# ---
## Test
test:
	$(eval DOCKER_IMAGE_TAG=${DOCKER_IMAGE_NAME}:${DOCKER_TAG})

	@echo "${DOCKER_IMAGE_TAG}"

## Bump major version number
bump_major:
	@echo "Bumping major version from ${CURRENT_VERSION}"
	bumpversion --current-version ${CURRENT_VERSION} major setup.py ${PROJECT_NAME}/__init__.py
	
## Bump minor version number
bump_minor:
	@echo "Bumping minor version from ${CURRENT_VERSION}"
	bumpversion --current-version ${CURRENT_VERSION} minor setup.py ${PROJECT_NAME}/__init__.py

## Bump patch version number
bump_patch:
	@echo "Bumping patch version from ${CURRENT_VERSION}"
	bumpversion --current-version ${CURRENT_VERSION} patch setup.py ${PROJECT_NAME}/__init__.py

## Build Python package
build:
	poetry build

## Check package build
check:
	twine check dist/*

## Publish to PyPI
publish: 
	poetry publish --username __token__ --password $PYPI_API_TOKEN

## Build Docker image
image:
	$(eval DOCKER_IMAGE_TAG=${DOCKER_IMAGE_NAME}:${DOCKER_TAG})

	@echo "Building docker image ${DOCKER_IMAGE_TAG}"
	docker build --build-arg BUILD_DATE=${BUILD_DATE} \
				--build-arg PROJECT_NAME=${PROJECT_NAME} \
				-t ${DOCKER_IMAGE_TAG} .
	@echo "Done"

## Git hooks
hooks:
	cp bin/post-checkout .git/hooks/post-checkout

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help build
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
