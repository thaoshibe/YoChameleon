#!/bin/bash
USER=thaon
WANDB_API_KEY="563710e55fec9aac8f27c7ab80cfed931a2096f5"
SCRIPT_DIR="/sensei-fs/users/thaon/code/YoChameleon"

# ============ 1. Experiment setup. ============
# To support automatic job submission.
if [ -z "$1" ]; then
    # If no job name is in argument, use the following one.
    JOB_NAME="shibe"
else
    JOB_NAME=$1
fi

CONFIG="llama_fusion_tiny_ldm_vae_imagenet.yaml"

# Change to your own repo root.
REPO_FOLDER="/sensei-fs/users/${USER}/Projects/chameleon1.5"

JOB_NAME_LEN=${#JOB_NAME}
# yyyy-MM-ddthh-mm-ss will be appended to the job name.
if (( $JOB_NAME_LEN > 44 )); then
    echo "JOB_NAME is too long ($JOB_NAME_LEN > 44). It should be less than 64 characters."
    exit
fi

# ============ 2. runai configs. ============
NUM_NODES=1

# RUNAI_PROJ=ilo-train-p4de
RUNAI_PROJ=ilo-noquota-p4de
RESEARCH_JACK_ID=6204
NODE_POOL=a100-80gb-1
DOCKER="docker-matrix-experiments-snapshot.dr-uw2.adobeitc.com/kineto:0.0.17-rc9"

# ============ 3. Submit the job. ============
DESTINATION_FOLDER=${REPO_FOLDER}
EXP_ROOT=${REPO_FOLDER}

runai submit --large-shm \
    -i $DOCKER \
    --backoff-limit 10 \
    --node-pools $NODE_POOL \
    --name $JOB_NAME \
    -g 8 \
    -p $RUNAI_PROJ \
    -l research_jack_id=$RESEARCH_JACK_ID \
    -l activity_type=focused_research \
    -l gpu-throttling-error-optout=true \
    -e USER=$USER \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e SCRIPT_DIR=$SCRIPT_DIR \
    --command -- bash -c "cd /sensei-fs/users/thaon/code/YoChameleon/; chmod +x launch_train.sh; umask 007; bash /sensei-fs/users/thaon/code/YoChameleon/launch_train.sh > /sensei-fs/users/thaon/output.log"
