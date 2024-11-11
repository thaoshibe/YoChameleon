#!/bin/bash
WANDB_API_KEY="563710e55fec9aac8f27c7ab80cfed931a2096f5"
SCRIPT_DIR="/sensei-fs/users/thaon/code/YoChameleon/scripts/sensei"

# ============ 1. Experiment setup. ============
# To support automatic job submission.
if [ -z "$1" ]; then
    # If no job name is in argument, use the following one.
    JOB_NAME="shibe"
else
    JOB_NAME=$1
fi
# Get current date in mm-dd format.
CURRENT_DATE=$(date +"%m%d-%H%M")
echo $CURRENT_DATE

# Append the current date (mm-dd) to the job name.
JOB_NAME="${JOB_NAME}-${CURRENT_DATE}"
echo $JOB_NAME

# ============ 2. Runai configs. ============

RUNAI_PROJ=ilo-train-p4de
# RUNAI_PROJ=ilo-noquota-p4de
RESEARCH_JACK_ID=6204
NODE_POOL=a100-80gb-1
# NODE_POOL=a100-80gb-2 ## This one is for noquota
DOCKER="docker-matrix-experiments-snapshot.dr-uw2.adobeitc.com/kineto:0.0.17-rc9"

# ============ 3. Submit the job. ============

runai submit --large-shm \
    -i $DOCKER \
    --node-pools $NODE_POOL \
    --name $JOB_NAME \
    -g 8 \
    -p $RUNAI_PROJ \
    -l research_jack_id=$RESEARCH_JACK_ID \
    -l activity_type=focused_research \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    --command -- bash -c '"cd '${SCRIPT_DIR}'; umask 007; bash ./api3.sh;"'