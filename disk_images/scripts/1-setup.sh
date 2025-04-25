#!/bin/bash

#rm packer_1.12.0_linux_arm64.zip
#rm packer
#rm LICENSE.txt
#wget https://releases.hashicorp.com/packer/1.12.0/packer_1.12.0_linux_arm64.zip
#unzip packer_1.12.0_linux_arm64.zip

#rm noble-server-cloudimg-arm64.img
#wget https://cloud-images.ubuntu.com/noble/20250403/noble-server-cloudimg-arm64.img
#qemu-img convert noble-server-cloudimg-arm64.img -O raw ./arm64.img.v2
#qemu-img resize -f raw arm64.img.v2 +20G

#rm disk_image_key*
#ssh-keygen -b 4096 -t rsa -f disk_image_key -q -N ""
cp shared/cloud_template.txt cloud.txt
echo "    ssh_authorized_keys:" >> cloud.txt
echo -n "      - " >> cloud.txt
cat disk_image_key.pub >> cloud.txt

cloud-localds --disk-format qcow2 cloud.img cloud.txt

