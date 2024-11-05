#!/bin/bash

# Define the project name
PROJECT="ilo-train-p4de"
# PROJECT="ilo-noquota-p4de"

# List of job names to delete
jobs=("api-1104-1130" "thao1105-9am" "thao1106-10am" "api2-1103-2335" "api3-1103-2334" "positive-only-1102-1004" "recog-1101-1751" "recog-1102-1355" "together-1101-1753" "together-1102-1018" "recog-1101-2337" "together-1101-2338")

# Loop through each job and delete it
for job in "${jobs[@]}"
do
  runai delete job "$job" -p "$PROJECT"
done

echo "All jobs deleted."

