#!/bin/bash

data=../data/text8

ncores=16 # set this to #logical cores of your machine (with hyper-threading if available)
niters=10


export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine

binary=../pWord2Vec
numactl --interleave=all $binary -train $data -output vectors.txt -size 100 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17

binary=../pSGNScc
numactl --interleave=all $binary -train $data -output vectors.txt -size 100 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17 -C 8 -T 500000

cp pWord2Vec_time pWord2Vec_text8
cp pSGNScc_time pSGNScc_text8

data=../data/1b
niters=1

binary=../pWord2Vec
numactl --interleave=all $binary -train $data -output vectors.txt -size 300 -window 5 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 11

binary=../pSGNScc
numactl --interleave=all $binary -train $data -output vectors.txt -size 300 -window 5 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 11 -C 8 -T 500000

cp pWord2Vec_time pWord2Vec_1b
cp pSGNScc_time pSGNScc_1b

echo ""
echo "Text8 dataset: Breakdown of Time"
awk '{print "\tMethod:\tTime/Epoch\tCreate InM\tCreate OutM\tSGD Time\tUpdate Min\tUpdate Mout\tOverhead"}' pWord2Vec_text8
awk '{print "\tpWord2Vec:\t"$2"\t"$6"\t\t"$8"\t\t"$4"\t\t"$10"\t\t"$12"\t\t"$14}' pWord2Vec_text8
awk '{print "\tpSGNScc:\t"$2"\t"$6"\t\t"$8"\t\t"$4"\t\t"$10"\t\t"$12"\t\t"$14}' pSGNScc_text8

echo "1B dataset: Breakdown of Time"
awk '{print "\tMethod:\tTime/Epoch\tCreate InM\tCreate OutM\tSGD Time\tUpdate Min\tUpdate Mout\tOverhead"}' pWord2Vec_1b
awk '{print "\tpWord2Vec:\t"$2"\t"$6"\t\t"$8"\t\t"$4"\t\t"$10"\t\t"$12"\t\t"$14}' pWord2Vec_1b
awk '{print "\tpSGNScc:\t"$2"\t"$6"\t\t"$8"\t\t"$4"\t\t"$10"\t\t"$12"\t\t"$14}' pSGNScc_1b

