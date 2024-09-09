#!/bin/bash

# run benchmarks
for e in `seq 5 18`
do
    echo $e
    CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE=0 python run_neuralmag.py $e &> out_e${e}
done

# cleanup summary
cat out_e* | grep NN= | awk 'BEGIN {print "#NN n_eval t_total throughput"} {print $2, $4, $6, $8}' | sort -n  | column -t > timings.dat

# plot results
python plot.py
