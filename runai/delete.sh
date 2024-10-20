#!/bin/bash

# Define the project name
# PROJECT="ilo-train-p4de"
PROJECT="ilo-noquota-p4de"

# List of job names to delete
jobs=(
  "yj-idea-1016-1821"
  "yj-idea-1015-1943"
  "yh-idea-1015-1851"
  "yj-idea-1015-1828"
  "smaller-lr-1014-2137"
  "test-1014-1659"
  "test-1014-1042"
  "test1-1014"
  "thao1020"
  "thao1023"
  "debug1019"
  "yj-idea-1016-1809"
  "yj-idea-quota-1016-1821"
)

# Loop through each job and delete it
for job in "${jobs[@]}"
do
  runai delete job "$job" -p "$PROJECT"
done

echo "All jobs deleted."

