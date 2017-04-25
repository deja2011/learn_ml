#!/bin/sh

wget http://spark-public.s3.amazonaws.com/neuralnets/Programming%20Assignments/Assignment2/assignment2.zip
unzip assignment2.zip
mv assignment2/* ./
rm -rf assignment2
rm -rf assignment2.zip
