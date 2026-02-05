#!/bin/bash
# Wrapper script to run vibrant with CUDA support
export LD_LIBRARY_PATH=/home/joe/code/vibrant/build/cuda:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
exec $(dirname "$0")/vibrant "$@"
