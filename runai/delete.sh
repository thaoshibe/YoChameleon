#!/bin/bash

# Define the project name
PROJECT="ilo-train-p4de"
# PROJECT="ilo-noquota-p4de"

# List of job names to delete
jobs=(
  "thao0111-9am"
  "a-gen-1030-2016"
  "a-recog-1030-2056"
  "disjoin-1028-1713"
  "neg1k-1028-1744"
  "neg1k-1029-2239"
  "neg1k-text-1028-1743"
  "neg1k-text-recog-1029-1829"
  "positive-1028-1800"
  "yollava-1029-1812"
)

# Loop through each job and delete it
for job in "${jobs[@]}"
do
  runai delete job "$job" -p "$PROJECT"
done

echo "All jobs deleted."

