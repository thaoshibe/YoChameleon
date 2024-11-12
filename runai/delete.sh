#!/bin/bash

# Define the project name
PROJECT="ilo-train-p4de"
# PROJECT="ilo-noquota-p4de"

# List of job names to delete
jobs=(
'train1-1111-1643'
'train1-1111-1647'
'train1-1111-1707'
'train1-1111-1717'
'train1-1111-1733'
'train2-1111-1739'
'train2-1111-1838'
'train2-1111-2157'
'sf2-1105-1347'

)

# Loop through each job and delete it
for job in "${jobs[@]}"
do
  runai delete job "$job" -p "$PROJECT"
done

echo "All jobs deleted."

