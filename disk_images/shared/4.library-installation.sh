#!/bin/bash

cd $HOME/libpickledevice
make -j`nproc`
./install-libpickledevice.sh
