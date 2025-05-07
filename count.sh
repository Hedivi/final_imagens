#!/bin/bash

for file in *;
do
  echo "Patient_$file:"	
  for i in 0 1 2 3 4 5;
  do
	  echo "$i: `find $file -iname *_$i.dcm | wc -l`;"
  done
  echo "?"
done
