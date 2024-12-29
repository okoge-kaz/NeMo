#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h=r7n4
#$ -l h_rt=2:8:00:00
#$ -o outputs/self_drain/$JOB_ID.log
#$ -e outputs/self_drain/$JOB_ID.log
#$ -p -5

./experiments/manual-drain/infinite_sleep.sh
