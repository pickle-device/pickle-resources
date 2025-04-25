#!/bin/bash

# m5 --addr is compatible with KVM

insmod /home/ubuntu/pickle_device_driver/build/pickle_driver.ko

if ! [[ "$IGNORE_M5" == 1 ]]; then
    m5 --addr readfile > script.sh
    if [ -s script.sh ]; then
        chmod +x script.sh
        echo "Executing the following commands"
        echo "//------"
        cat script.sh
        echo "//------"
        #echo "exit 1"
        #m5 --addr exit
        #sleep 1
        ./script.sh
        #sleep 1
        m5 --addr workend 0 1
        #m5 --addr exit
    else
        IGNORE_M5=1 /bin/bash
    fi
fi
