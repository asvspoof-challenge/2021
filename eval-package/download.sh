#!/bin/bash
# This bash script only works for Linux Ubuntu.
# If script does not work, please download the file manually

URLLINKS=(https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz
	  https://www.asvspoof.org/asvspoof2021/PA-keys-full.tar.gz
          https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz)

MD5SUMVALS=(037592a0515971bbd0fa3bff2bad4abc
	    a639ea472cf4fb564a62fbc7383c24cf
	    dabbc5628de4fcef53036c99ac7ab93a)

if command -v wget &> /dev/null
then
    TOOL="wget -q --show-progress"
else
    echo "Cannot find a tool to download the file. Please manually download it"
    for URLLINK in ${URLLINKS}
    do
	echo ${URLLINK}
    done
    exit
fi

if command -v md5sum &> /dev/null
then
    MD5TOOL="md5sum"
else
    MD5TOOL=""
fi


# download
for idx in 0 1 2
do
    URLLINK=${URLLINKS[${idx}]}
    MD5VAL=${MD5SUMVALS[${idx}]}
    
    PACKNAME=$(basename ${URLLINK})

    echo "Download ${URLLINK}"
    ${TOOL} ${URLLINK}
    
    while [ ! -e "${PACKNAME}" ];
    do
	echo "File server is busy. Re-try to download ${URLLINK}"
	${TOOL} ${URLLINK}
	sleep 0.5
    done
    
    if [ ! -e "${PACKNAME}" ];
    then
	echo "Failed to download the file."
	echo "Please manully download ${URLLINK}."
	echo "File server may be busy."
	echo "Please try multiple times"
	exit
    else
	if [ -z "$MD5TOOL" ];
	then
	    echo "Cannot a find a tool for checksum."
	    echo "checksum of ${PACKNAME} should be ${MD5VAL}"
	    tar -xzf ${PACKNAME}
	else
	    md5sum=$(${MD5TOOL} ${PACKNAME} | cut -d ' ' -f 1)
	    if [ "${md5sum}" == "${MD5VAL}" ];
	    then
		tar -xzf ${PACKNAME}
            else
		echo "Downloaded file seems to be damaged"
		echo "Please contact with the organizers"
	    fi
	fi
    fi
done
