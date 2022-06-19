#!/bin/bash

# Run preprocessing the collate data according to dygiepp

# radgraph
echo "Processing RadGraph.."
python scripts/prepare_data_radgraph.py --DATA_DIR radgraph
python scripts_dygiepp/collate.py radgraph/processed JSON_CACHED/radgraph
echo "RadGraph dataset ready."