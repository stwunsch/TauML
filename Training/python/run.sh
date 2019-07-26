#!/bin/bash


for FILE in tau_HTT.h5 e_DY.h5 mu_DY.h5 jet_DY.h5
do
    python analyze_gradients.py --input ../../../data/${FILE}.h5 --model ../../../model/DeepTau2017v2p6_step1_e6.pb --output .
done
