#!/usr/bin/env bash
if [[ "$EUID" -ne 0 ]]
  then echo "fatal error: run as root to install baylib"
  exit
fi

echo "Installing baylib..."

cd $(dirname "$0")/..

# compile
mkdir build
cd build
cmake ..
make

# install
sudo make install
