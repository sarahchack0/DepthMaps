#!/bin/bash


#SBATCH -J train_bert_svm
#SBATCH --mem=40G
#SBATCH -o question_weighted_%j.txt
#SBATCH -e question_weighted_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=100:00:00


python simplified_train.py

