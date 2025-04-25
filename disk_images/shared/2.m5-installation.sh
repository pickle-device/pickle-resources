#!/bin/bash

cd $HOME
git clone https://github.com/gem5/gem5
cd gem5
git checkout v24.1.0.2
cd $HOME/gem5/util/m5
/home/ubuntu/.local/bin/scons build/arm64/out/m5

sh -c "echo \"/home/ubuntu/gem5-init.sh\" >> /home/ubuntu/.bashrc"
