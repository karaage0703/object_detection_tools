#!/bin/bash
i=0
for f in ./train/*.tfrecord

do
	g=./train/frame`printf %04d $i`.tfrecord
	echo $f
	echo $g
	mv $f $g
	i=$((i+1))
done

i=0

for f in ./val/*.tfrecord
do
	g=./val/frame`printf %04d $i`.tfrecord
	echo $f
	echo $g
	mv $f $g
	i=$((i+1))
done
