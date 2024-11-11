#!/bin/bash

# Define the project name
PROJECT="ilo-train-p4de"
# PROJECT="ilo-noquota-p4de"

# List of job names to delete
jobs=(
    "thao1114-9am"
    "thao1112-11pm"
    "thao1111-2pm"
    "sf3-1105-1611"
    "sf2-1105-1609"
    "sf2-1105-1347"
    "sf3-1105-1205"
    "sf2-1105-1204"
    "sfrecog-1105-0745"
    "api3-1104-2359"
    "selfprompting-recog-1104-2357"
    "api3-1104-1937"
    "api3-1104-1924"
    "thao1109-10am"
)

# Loop through each job and delete it
for job in "${jobs[@]}"
do
  runai delete job "$job" -p "$PROJECT"
done

echo "All jobs deleted."

