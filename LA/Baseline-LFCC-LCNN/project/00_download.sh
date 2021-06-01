#!/bin/bash

# Download toy data package and pre-trained models
echo "=== Downloading toy dataset ==="
SRC=https://www.dropbox.com/sh/gf3zp00qvdp3row/AABc-QK2BvzEPj-s8nBwCkMna/temp/asvspoof2021/toy_example.tar.gz
cd DATA
wget -q ${SRC}
cd ..

echo "=== Downloading pre-trained models ==="
if [ -d baseline_DF ];
then
    SRC=https://www.asvspoof.org/asvspoof2021/pre_trained_DF_LFCC-LCNN.zip
    wget -q ${SRC}
    if [ -f pre_trained_DF_LFCC-LCNN.zip ];
    then
	unzip pre_trained_DF_LFCC-LCNN.zip
    else
	SRC=https://www.dropbox.com/sh/gf3zp00qvdp3row/AABg4OUDKFvFonI-HmIzT5qIa/temp/asvspoof2021/df_trained_network.pt
	wget -q ${SRC}
    fi
    mv df_trained_network.pt baseline_DF/__pretrained/trained_network.pt
fi

if [ -d baseline_LA ];
then
    SRC=https://www.asvspoof.org/asvspoof2021/pre_trained_LA_LFCC-LCNN.zip
    wget -q ${SRC}
    if [ -f pre_trained_LA_LFCC-LCNN.zip ];
    then
	unzip pre_trained_LA_LFCC-LCNN.zip
    else
	SRC=https://www.dropbox.com/sh/gf3zp00qvdp3row/AADWOs9cmLBzWuRPbh5m10YVa/temp/asvspoof2021/la_trained_network.pt
	wget -q ${SRC}
    fi
    mv la_trained_network.pt baseline_LA/__pretrained/trained_network.pt
fi

if [ -d baseline_PA ];
then
   SRC=https://www.asvspoof.org/asvspoof2021/pre_trained_PA_LFCC-LCNN.zip
   wget -q ${SRC}
   if [ -f pre_trained_PA_LFCC-LCNN.zip ];
   then
       unzip pre_trained_PA_LFCC-LCNN.zip
   else
       SRC=https://www.dropbox.com/sh/gf3zp00qvdp3row/AACOBr0ymqsMA0rKctMxkKaxa/temp/asvspoof2021/pa_trained_network.pt
       wget -q ${SRC}
   fi
   mv pa_trained_network.pt baseline_PA/__pretrained/trained_network.pt
fi

echo "Download done"
