#!/bin/bash -
#===============================================================================
#
#          FILE: test_batch.sh
#
#         USAGE: ./test_batch.sh batch_dir batch_name no_processes
#
#   DESCRIPTION: test a whole batch. Not robust. WILL break probably
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: Oliver Tautz
#  ORGANIZATION:
#       CREATED: 06/22/2021 08:03:49 PM
#      REVISION:  ---
#===============================================================================

set -o nounset                                  # Treat unset variables as an error

batch_dir=$1
batch_name=$2

to_test=$(find $1 -type d -name "$2*")
for f in $to_test
 do
      python test.py $f --save-vids --sequence-len 100 --max-steps 1000 --no-cpu 8 --num-threads 2
      wait -n
      echo LOLOLOL

done