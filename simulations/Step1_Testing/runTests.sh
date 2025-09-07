#!/bin/bash
for Model in "logistic" "normal" "poisson" "logcosh"
do 
    for hypothesis in "H0" "H1"
    do
        python3 runTests.py --Model $Model --hypothesis $hypothesis
    done
done
