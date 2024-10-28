#!/bin/bash

# Define the project name
PROJECT="ilo-train-p4de"
# PROJECT="ilo-noquota-p4de"

# List of job names to delete
jobs=(
    "disjoin-1027-1710"
    "disjoin-1027-1659"
    "disjoin-1027-1658"
    "disjoin-1027-1658"
    "e1000-1026-1242"
    "neg1k-text-1026-1218"
    "neg1k-text-1026-1201"
    "lr0005-1026-1057"
    "neg1k-text-1026-1035"
    "positive-1025-2357"
    "neg1k-1025-2349"
    "neg1k-text-recog-1025-2334"
    "thao1029-10am"
    "thao1029-9am"
    "negative2000-text-1021-1241"
    "negative1000-text-recog-1021-1212"
    "negative1000-text-1021-1211"
    "yollava-setting-1020-1409"
    "prefix-positive-only-1020-1328"
    "thao1024"
)



# Loop through each job and delete it
for job in "${jobs[@]}"
do
  runai delete job "$job" -p "$PROJECT"
done

echo "All jobs deleted."

