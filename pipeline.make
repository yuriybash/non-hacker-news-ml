SHELL=/usr/bin/env bash -o pipefail

# Fixed paths to data, scripts, etc.:
BASE=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
DATA=$(BASE)/data
FEATURES=$(BASE)/features
DATA_SCRIPTS=$(BASE)/scripts/data
MODEL_SCRIPTS=$(BASE)/scripts/model
MODELS=$(BASE)/models
CFG=$(BASE)/config.yml

# FLAGS
SAVE_FEATURES="--save-features"
SAVE_MODEL="--save-model"

# Listing of all rules in this makefile:
.PHONY: format_data prepare_features train test clean

# Parse data, train model
# TODO: missing step of choosing best model from "results" dir
.DEFAULT_GOAL: all
all: clean format_data compare_models train_model

# The 'format_data' rule converts our raw data into cleaned, formatted data.
format_data:
	echo "FORMATTING DATA"
	python $(DATA_SCRIPTS)/raw_json_to_stripped_csv

# Trains and tests various models using GridSearchCV and a config file
compare_models:
	echo "RUNNING GRID SEARCH using grid_config.yml"
	python $(MODEL_SCRIPTS)/grid_search.py


# Save feature vectorizers
prepare_features:
    echo "TRAINING MODEL, SAVING FEATURES"
	python $(MODEL_SCRIPTS)/train_model.py $(SAVE_FEATURES)

train_model:
    echo "TRAINING MODEL, SAVING MODEL"
    python $(MODEL_SCRIPTS)/train_model.py $(SAVE_MODEL)

# The 'train' rule trains a model on our features that we've just generated.
train:
	echo "TRAINING MODEL, SAVING MODEL AND FEATURES"
	python $(MODEL_SCRIPTS)/train_model.py $(SAVE_MODEL) $(SAVE_FEATURES)

# Clean everything up by deleting detritus.
clean:
	echo "CLEANING..."
	find . -name "*.pyc" -exec rm -f {} \;
