#!/bin/bash
########################
# Script for toy_example
# This script will 
#  1. install pytorch env using conda 
#  2. untar toy data set
#  3. run evaluation and training process
#
# If GPU memory is less than 16GB, please reduce
#   --batch-size in 00_train.sh
########################
RED='\033[0;32m'
NC='\033[0m'

echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step1. install conda environment${NC}"

# create conda environment
bash conda.sh

# untar the toy-example data
echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step2. untar toy data set${NC}"

cd DATA
tar -xzf toy_example.tar.gz
cd ..

echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step3. run evaluation process (using pre-trained model)${NC}"
# run scripts

cd baseline_LA
# evaluation using pre-trained model
bash 01_eval.sh

echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step4. run training process (start with pre-trained model)${NC}"
#training, with pre-trained model as initialization
bash 00_train.sh

