#!/bin/sh

code=../hyperwords
if [ "$1" != "" ]; then
	src_model=$1
else
	src_model=vectors.txt
fi
model=temp

cp $src_model $model.words

python $code/hyperwords/text2numpy.py $model.words

echo "WS353 Results"
echo "-------------"
python $code/hyperwords/ws_eval.py embedding $model $code/testsets/ws/ws353.txt
echo

echo "Google Analogy Results"
echo "----------------------"
python $code/hyperwords/analogy_eval.py embedding $model $code/testsets/analogy/google.txt
echo

