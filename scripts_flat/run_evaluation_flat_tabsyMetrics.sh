#!/bin/bash
# calculate quality metrics using tabsyn repo https://github.com/amazon-science/tabsyn

# Define the dataset name (replace with the actual dataset name)
DATANAMES=("adult" "acs-income")

# List of models to loop over
MODELS=("mostly" "tabddpm" "tabsyn" "stasy")

# Loop over each model
for METHOD_NAME in "${MODELS[@]}"
do
	for DATANAME in "${DATANAMES[@]}"
	do
	    # Construct the path to the synthetic data
	    PATH_TO_SYNTHETIC_DATA="synthetic/${DATANAME}/${METHOD_NAME}.csv"
	    
	    echo "Running eval for $METHOD_NAME with data at $PATH_TO_SYNTHETIC_DATA"
	    python eval/eval_density.py --dataname "$DATANAME" --model "${METHOD_NAME}" --path "$PATH_TO_SYNTHETIC_DATA"
	    python eval/eval_quality.py --dataname "$DATANAME" --model "${METHOD_NAME}" --path "$PATH_TO_SYNTHETIC_DATA"   
	done
done
