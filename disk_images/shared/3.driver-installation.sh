#!/bin/bash

cd $HOME
rm -rf linux-6.6.71.*
rm -rf linux_src
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.6.71.tar.xz
tar xf linux-6.6.71.tar.xz
mv linux-6.6.71 linux_src
cd linux_src
wget https://raw.githubusercontent.com/gem5/gem5-resources/f32e4ccd6cfc197699bc58f2f64b139ad95cc830/src/linux-kernel/linux-configs/config.arm64.5.15.36
mv config.arm64.5.15.36 .config
yes "" | make oldconfig
make -j$(nproc)
make prepare -j$(nproc)
make modules_prepare -j$(nproc)

cd $HOME/pickle_device_driver
./compile-driver.sh
./install-module.sh
