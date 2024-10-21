#!/bin/bash

# Define the project name
# PROJECT="ilo-train-p4de"
PROJECT="ilo-noquota-p4de"

# List of job names to delete
jobs=(
  "yollava-setting-1020-1403"
  "yollava-setting-1020-1350"
  "yollava-setting-1020-1343"
  "yollava-setting-1020-1334"
  "yollava-setting-1020-1329"
  "prefix-positive-only-1020-1310"
  "prefix-positive-only-1020-1248"
)

# Loop through each job and delete it
for job in "${jobs[@]}"
do
  runai delete job "$job" -p "$PROJECT"
done

echo "All jobs deleted."

