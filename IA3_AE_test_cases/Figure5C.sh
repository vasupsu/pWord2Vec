#!/bin/bash

data=../data/1b

niters=1

rm word2vec_5C pWord2Vec_5C pSGNScc_5C threadID
for i in 1 4 8 16 #i is the number of threads
do
	echo "Threads" $i
	echo $i >> threadID
	ncores=$i # set this to #logical cores of your machine (with hyper-threading if available)
	binary=../word2vec
	numactl --interleave=all $binary -train $data -output vectors.txt -size 300 -window 5 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0
	cat word2vec_time >> word2vec_5C

	binary=../pWord2Vec
	numactl --interleave=all $binary -train $data -output vectors.txt -size 300 -window 5 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 11
	cat pWord2Vec_time >> pWord2Vec_5C

	binary=../pSGNScc
	numactl --interleave=all $binary -train $data -output vectors.txt -size 300 -window 5 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 11 -C 8 -T 500000
	cat pSGNScc_time >> pSGNScc_5C
done
echo "1B dataset: Time Per Epoch"

echo ""
awk '{print "Threads\tWord2Vec\tpWord2Vec\tpSGNScc"}' word2vec_time
paste threadID word2vec_5C pWord2Vec_5C pSGNScc_5C | awk '{print $1"\t"$2"\t"$5"\t"$19}'
