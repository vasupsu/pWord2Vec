#!/bin/bash

data=../data/text8

ncores=16 # set this to #logical cores of your machine (with hyper-threading if available)
niters=10


export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
binary=../word2vec
numactl --interleave=all $binary -train $data -output vectors.txt -size 100 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec
numactl --interleave=all $binary -train $data -output vectors.txt -size 100 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17

binary=../pSGNScc
numactl --interleave=all $binary -train $data -output vectors.txt -size 100 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17 -C 8 -T 500000

echo "Text8 dataset: Time Per Epoch"
awk '{print "\tWord2Vec:\t" $1/10}' word2vec_time
awk '{print "\tpWord2Vec:\t" $2/10}' pWord2Vec_time
awk '{print "\tpSGNScc:\t" $2/10}' pSGNScc_time

data=../data/1b
niters=1

binary=../word2vec
numactl --interleave=all $binary -train $data -output vectors.txt -size 300 -window 5 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec
numactl --interleave=all $binary -train $data -output vectors.txt -size 300 -window 5 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 11

binary=../pSGNScc
numactl --interleave=all $binary -train $data -output vectors.txt -size 300 -window 5 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 11 -C 8 -T 500000

echo "1B dataset: Time Per Epoch"
awk '{print "\tWord2Vec:\t" $1}' word2vec_time
awk '{print "\tpWord2Vec:\t" $2}' pWord2Vec_time
awk '{print "\tpSGNScc:\t" $2}' pSGNScc_time

