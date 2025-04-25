#!/bin/bash

sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y install build-essential git python3-pip gfortran cmake ninja-build git-lfs libomp-dev
sudo apt-get -y install m4 scons zlib1g zlib1g-dev \
    libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
    python3-dev doxygen libboost-all-dev libhdf5-serial-dev python3-pydot \
    libpng-dev libelf-dev pkg-config pip python3-venv black wget \
    cmake gdb \
    libncurses-dev gawk flex bison openssl libssl-dev dkms libelf-dev \
    libudev-dev libpci-dev libiberty-dev autoconf llvm bc mold
pip install scons --user

# Removing snapd
sudo snap remove $(snap list | awk '!/^Name|^core/ {print $1}') # not removing the core package as others depend on it
sudo snap remove $(snap list | awk '!/^Name/ {print $1}')
sudo systemctl disable snapd.service
sudo systemctl disable snapd.socket
sudo systemctl disable snapd.seeded.service
sudo apt remove --purge -y snapd
sudo apt -y autoremove
sudo apt -y autopurge

# Removing mounting /boot/efi
sudo sed -i '/\/boot\/efi/d' /etc/fstab
sudo systemctl stop boot-efi.mount
sudo systemctl disable boot-efi.mount

# Removing cloud-init
sudo touch /etc/cloud/cloud-init.disabled

sudo mv /home/ubuntu/serial-getty@.service /lib/systemd/system/
