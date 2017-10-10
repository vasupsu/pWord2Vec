#!/bin/bash

data=../data/1b

niters=1
ncores=16 # set this to #logical cores of your machine (with hyper-threading if available)
binary=../pSGNScc

#rm Cvalues table_3_C
for C in 16 
do
	echo "C " $C
	echo $C >> Cvalues

	numactl --interleave=all $binary -train $data -output vectors.txt -size 300 -window 5 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 11 -C $C -T 500000
	cat pSGNScc_time >> table_3_C
done
echo "1B dataset: Varying C"

echo ""
awk '{print "C\tTime/Epoch\tSGD Computation\tOverhead"}' pSGNScc_time
paste Cvalues table_3_C | awk '{print $1"\t"$3"\t"$5"\t\t"$15}'
