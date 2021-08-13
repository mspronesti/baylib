#!/usr/bin/env bash
if [[ "$EUID" -ne 0 ]]
  then echo "fatal error: run as root to install baylib"
  exit
fi

# compile
cd ..
mkdir -p build
cd build
cmake ..
make

# install
sudo make install
