#!/bin/bash

# This script only works for Mac and Linux.
# If curl or wget is not available, please download the file manually

URLLINK=https://www.asvspoof.org/resources/LA-keys-stage-1.tar.gz
PACKNAME=LA-keys-stage-1.tar.gz

if command -v wget &> /dev/null
then
    TOOL="wget -q --show-progress"
    ${TOOL} ${URLLINK}
elif command -v curl &> /dev/null
then
    TOOL="curl -L -O -J"
    ${TOOL} ${URLLINK}
else
    echo "Could not find a tool to download files"
    echo "Please manully download the file from ${URLLINK}"
    echo "Then please untar it. You should get a folder called ./keys"
    exit
fi

if [ ! -e "${PACKNAME}" ];
then
    echo "Could not automatically download the file"
    echo "Please manully download the file from ${URLLINK}"
    echo "Then please untar it. You should get a folder called ./keys"
    exit
else
    tar -xzvf ${PACKNAME}
fi

