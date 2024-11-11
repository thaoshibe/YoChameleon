# README for Yuheng Li

#### Before you run

```
# 1. Copy dataset from my sensei to your sensei

mkdir /sensei-fs/users/yuhli/data/
cp /sensei-fs/users/thaon/data/yochameleon-data.zip /sensei-fs/users/yuhli/data/

# 2. Clone the code

mkdir /sensei-fs/users/yuhli/code/
git clone git@github.com:thaoshibe/YoChameleon.git

# (or if you already has the repo)

cd /sensei-fs/users/yuhli/code/YoChameleon
git pull origin main


# 3. Change the config file [savedir: '/sensei-fs/users/thaon/ckpt/'](https://github.com/thaoshibe/YoChameleon/blob/main/config/selfprompting.yaml#L26) to the folder that you want

```

#### Now, you can run the code from your local

```
# runai submit script is located at ./runai

#
# cd runai
# 
# this script will create job-name `train1-DDMM-HHMM` name, and run train1.sh file
bash submit_train_job_yuhli.sh train1

# You need to submit 10 jobs, to train train1-train10.sh!!!
bash submit_train_job_yuhli.sh train1
bash submit_train_job_yuhli.sh train2
bash submit_train_job_yuhli.sh train3
bash submit_train_job_yuhli.sh train4
bash submit_train_job_yuhli.sh train5
bash submit_train_job_yuhli.sh train6
bash submit_train_job_yuhli.sh train7
bash submit_train_job_yuhli.sh train8
bash submit_train_job_yuhli.sh train9
bash submit_train_job_yuhli.sh train10
```

Thank you, hopefully it will work. 
