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
no_processes=$3

to_test=$(find $1 -type d -name "$2*")
for f in $to_test
 do
    if [ $no_processes -gt 0 ]; then
      echo python test.py $f --save-vids
      #this should be easier ... --
      no_processes=$((no_processes - 1))

    else

      wait
      no_processes=3
      echo python test.py $f --save-vids
      #this should be easier ... --
      no_processes=$((no_processes - 1))

    fi

done