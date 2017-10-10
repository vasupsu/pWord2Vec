#!/bin/bash

data=../data/1b

niters=1
ncores=16 # set this to #logical cores of your machine (with hyper-threading if available)
binary=../pSGNScc

rm Tvalues table_3_T
for T in 100000 500000 1000000 2000000 
do
	echo "T " $T
	echo $T >> Tvalues

	numactl --interleave=all $binary -train $data -output vectors.txt -size 300 -window 5 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 11 -C 8 -T $T
	cat pSGNScc_time >> table_3_T
done
echo "1B dataset: Varying T"

echo ""
awk '{print "T\tTime/Epoch\tOverhead"}' pSGNScc_time
paste Tvalues table_3_T | awk '{print $1"\t"$3"\t\t"$15}'
