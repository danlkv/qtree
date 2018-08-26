#!/usr/bin/env bash
echo Hello world

mkdir data
python3.6 speed_measure.py
for nproc in {2..5}
do
		mpiexec -n "$nproc" python3.6 speed_measure_parallel.py
done

