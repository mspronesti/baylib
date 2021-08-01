#!/bin/bash

sudo apt update
sudo apt install build-essential

# CMake
sudo apt install cmake -y

# Boost
sudo apt install libboost-all-dev

# openCL
sudo apt install ocl-icd-opencl-dev
sudo apt install mesa-opencl-icd

# google test
git clone https://github.com/google/googletest
cd googletest
mkdir build
cd build
cmake ..
make
sudo make install
