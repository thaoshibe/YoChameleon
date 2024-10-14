#!/bin/bash
cd /sensei-fs/users/thaon/code/YoChameleon/
umask 007
bash launch_train.sh > /sensei-fs/users/thaon/output.log 2>&1
