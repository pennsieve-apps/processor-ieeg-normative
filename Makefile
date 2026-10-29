.PHONY: help test fixture run run-demo validate-atlas atlas-subjects check-vendored clean

SERVICE_NAME ?= processor-ieeg-normative
PYTHON ?= python3
IMAGE ?= processor-ieeg-normative:dev
WORK ?= ../workflow-epilepsy-ieeg/data/work
ATLAS ?= ../workflow-epilepsy-ieeg/data/atlas/HUP_atlas_final.mat

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make test           - pytest: synthetic atlas + features + channels through the processor (no Docker)"
	@echo "make fixture        - write the synthetic fixture to data/fixture/linked/{features,ieeg,reference}/"
	@echo "make run            - build the image and run it in Docker over data/fixture/linked/, writing to data/output/"
	@echo "make run-demo       - run the image over WORK=$(WORK) (bandpower + prep/ieeg) and ATLAS into data/output-demo/"
	@echo "make validate-atlas - leave-one-patient-out AUCs over every atlas patient (ATLAS=$(ATLAS))"
	@echo "make atlas-subjects - atlas patients 1-5 as an ieeg-prep-style derivative in data/atlas-subjects/ieeg/"
	@echo "make check-vendored - processor/spectral.py must equal ../processor-ieeg-bandpower/processor/spectral.py"
	@echo "make clean          - remove data/output*/ and data/fixture/"
	@echo ""
	@echo "PYTHON=$(PYTHON) (set PYTHON=.venv/bin/python after pip install -r requirements-dev.txt)"

test:
	$(PYTHON) -m pytest -q tests

fixture:
	$(PYTHON) -m tests.make_fixture data/fixture

run: clean fixture
	mkdir -p data/output
	docker compose -f docker-compose.yml down --remove-orphans
	docker compose -f docker-compose.yml build
	docker compose -f docker-compose.yml up --exit-code-from processor
	docker compose -f docker-compose.yml down --remove-orphans
	@echo "--- outputs:"
	@find data/output -type f | sort

run-demo:
	docker build --platform linux/amd64 -t $(IMAGE) .
	rm -rf data/output-demo && mkdir -p data/output-demo
	docker run --rm --platform linux/amd64 --memory 2g --env-file dev.env \
		-v "$(abspath $(WORK))/bandpower":/data/input/features:ro \
		-v "$(abspath $(WORK))/prep/ieeg":/data/input/ieeg:ro \
		-v "$(abspath $(ATLAS))":/data/input/reference/HUP_atlas_final.mat:ro \
		-v "$(CURDIR)/data/output-demo":/data/output $(IMAGE)

validate-atlas:
	mkdir -p data
	$(PYTHON) -m scripts.validate_atlas $(ATLAS) --out data/validation-atlas.csv

atlas-subjects:
	$(PYTHON) -m scripts.atlas_subjects $(ATLAS) data/atlas-subjects/ieeg --patients 1,2,3,4,5

check-vendored:
	cmp processor/spectral.py ../processor-ieeg-bandpower/processor/spectral.py && echo "spectral.py: identical"

clean:
	rm -rf data/output data/output-demo data/fixture
