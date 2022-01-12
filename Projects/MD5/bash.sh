#!/bin/bash

for (( device=1; device<=4; device++ ))
do
	filePathThreads="./csv/"$device"data.csv"
	echo $filePathThreads
	echo 'char;TIME;' > $filePathThreads
	word="z"

	for (( wordLen=1; wordLen<=8; wordLen++ ))
	do
		./main $wordLen `echo -n $word | md5sum` 1000 1024 $device 0 >> $filePathThreads
		word+=z
	done
done
