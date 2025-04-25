#!/bin/bash

eval `ssh-agent -s`
ssh-add disk_image_key
#ssh -p 5556 ubuntu@localhost
