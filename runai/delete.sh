#!/bin/bash

# Define the project name
PROJECT="ilo-train-p4de"
# PROJECT="ilo-noquota-p4de"

# List of job names to delete
jobs=(
    "neg1k-text-extraepoch-1024-1019"
    "neg1k-text-extraepoch-1024-1018"
    "neg1k-text-extraepoch-1024-1016"
    "neg1k-text-extraepoch-1024-0958"
)



# Loop through each job and delete it
for job in "${jobs[@]}"
do
  runai delete job "$job" -p "$PROJECT"
done

echo "All jobs deleted."

