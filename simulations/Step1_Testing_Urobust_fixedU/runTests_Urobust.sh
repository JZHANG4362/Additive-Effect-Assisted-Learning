#!/bin/bash
for Model in "logistic" "poisson" "normal" "logcosh" 
do 
    for hypothesis in "H0" "H1"
    do
        python3 runTests_Urobust.py --Model $Model --hypothesis $hypothesis
    done
done
