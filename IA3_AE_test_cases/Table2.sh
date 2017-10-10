#!/bin/bash

data=../data/text8

ncores=16 # set this to #logical cores of your machine (with hyper-threading if available)
niters=10

export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
binary=../word2vec
numactl --interleave=all $binary -train $data -output Word2Vec_text8.txt -size 100 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec
numactl --interleave=all $binary -train $data -output pWord2Vec_text8.txt -size 100 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17

binary=../pSGNScc
numactl --interleave=all $binary -train $data -output pSGNScc_text8.txt -size 100 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17 -C 8 -T 500000

data=../data/1b
niters=5

binary=../word2vec
numactl --interleave=all $binary -train $data -output Word2Vec_1b.txt -size 300 -window 5 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec
numactl --interleave=all $binary -train $data -output pWord2Vec_1b.txt -size 300 -window 5 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 11

binary=../pSGNScc
numactl --interleave=all $binary -train $data -output pSGNScc_1b.txt -size 300 -window 5 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 11 -C 8 -T 500000

word2vec_text8score=`sh eval.sh Word2Vec_text8.txt  | awk '{if ($1=="Word")score[$2]=$3} END {print "Word2ec\t"score["Similarity"]"\t\t"score["Analogy"]}'`
pword2vec_text8score=`sh eval.sh pWord2Vec_text8.txt  | awk '{if ($1=="Word")score[$2]=$3} END {print "pWord2Vec\t"score["Similarity"]"\t\t"score["Analogy"]}'`
psgnscc_text8score=`sh eval.sh pSGNScc_text8.txt  | awk '{if ($1=="Word")score[$2]=$3} END {print "pSGNScc\t"score["Similarity"]"\t\t"score["Analogy"]}'`
word2vec_1bscore=`sh eval.sh Word2Vec_1b.txt  | awk '{if ($1=="Word")score[$2]=$3} END {print "Word2ec\t"score["Similarity"]"\t\t"score["Analogy"]}'`
pword2vec_1bscore=`sh eval.sh pWord2Vec_1b.txt  | awk '{if ($1=="Word")score[$2]=$3} END {print "pWord2Vec\t"score["Similarity"]"\t\t"score["Analogy"]}'`
psgnscc_1bscore=`sh eval.sh pSGNScc_1b.txt  | awk '{if ($1=="Word")score[$2]=$3} END {print "pSGNScc\t"score["Similarity"]"\t\t"score["Analogy"]}'`
echo "text8 dataset"
echo $word2vec_text8score
echo $pword2vec_text8score
echo $psgnscc_text8score
echo "1B dataset"
echo $word2vec_1bscore
echo $pword2vec_1bscore
echo $psgnscc_1bscore
