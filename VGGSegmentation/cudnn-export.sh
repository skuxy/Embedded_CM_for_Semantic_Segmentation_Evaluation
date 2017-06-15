#!/bin/bash
#

LD_LIBRARY_PATH="/opt/libs/cudnn-8.0-linux-x64-v5.1/lib64/" python train.py
#export LD_LIBRARY_PATH="/opt/libs/cudnn-8.0-linux-x64-v5.1/lib64/":$LD_LIBRARY_PATH
echo "exported!"
