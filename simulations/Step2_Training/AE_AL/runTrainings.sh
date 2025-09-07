#!/bin/bash
for Model in "logistic" "normal" "poisson" "logcosh"
do 
    python3 runTrainings.py --Model $Model 
done
