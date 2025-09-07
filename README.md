**Thank you for trying our codes.**

## Sections 19.1.1 and 19.1.2 (S1 testing for logistic regression) of the Supplement:

The codes are stored in the folder `simulations/Step1_Testing`

- Run `runTests.sh` to obtain experiment results.

- Run `graphsH0.ipynb` to generate Q-Q plots (**Figure 12** in the supplement).

- Run `tableH0_rejectionrates.ipynb` to generate a table of rejection rates under H0 (**Table 4** in the supplement).

- Run `graphsH1.ipynb` to plot rejection rates under H1 (**Figure 13** in the supplement). 

## Sections 19.1.3 (S2 training for logistic regression) of the Supplement:

- run `simulations/Step2_Training/AE_AL/runTrainings.sh`, `simulations/Step2_Training/VFL/trainingVFL.py`, and `simulations/Step2_Training/Plot.ipynb`.

  (Note: trainingVFL.py needs to be run multiple times with settings {Q=5, 10, or 25, mu =0, or 0.1} and {Q = 1, mu = 0})  

- `Run Plot.ipynb` (**Figure 13** in the supplement).

  
## Sections 19.2 (Other models) of the Supplement:
To generate `Figures 15-23`, for the files from the above step,
change the argument `Model = "logistic"`  in trainingVFL.py to `Model = "normal"`,  `Model = "poisson"`,  and `Model = "logcosh"` and repeat the
above different settings of Q and mu to generate simulation results for different models.

## Section 4.1 of the main text:

The codes are stored in the folder `MIMIC3`. 
Sequentially run: `test1_dataProcessing.ipynb`, `test2_testing.py` (reproduces the testing result), `test3_data_splitting.ipynb`, `test4_AE_AL.ipynb`, `test5_VFL_tuning.ipynb`, `test6_Plot.ipynb` (reproduces the training result in **Figure 4** Hospital Length of Stay)

## Section 4.2 of the main text:

The codes are stored in the folder `car_data`.
test2_testing.py (reproduces the testing result)
test3_carPlots.py (reproduces **Figure 5**)
To reproduce the training result (**Figure 4** Car Top Speeds), sequentially run: test4_data_splitting.ipynb,
 test5_AE_AL.ipynb, test6_VFL_tuning.ipynb, test7_Plot.ipynb 
To reproduce the preprocessed data, run: test1_transfer_1.py, test1_transfer_2.py (Remember to download the dataset from
(http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html), put the folder `compcars` in the `data` folder before running the codes.
Make sure to follow the steps in https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt.
For Mac or Linux, you may need to zip and then unzip a file.)

## Section 4.3 of the main text:

The codes are stored in the folder `kddcup99-mld`.
Sequentially run: test1_dataProcessing.ipynb, test2_testing.py (reproduces the testing result), test3_data_splitting.ipynb, test4_AE_AL.ipynb, and test6_Plot.ipynb (reproduces the training result in Figure DoS Attack).

## Section 13.2 of the Supplement:

The codes are stored in `simulations/Step1_Testing/ridgeTesting` and `simulations/Step2_Training/ridgeFitting`.
To reproduce **Figures 1-5**, run ridge_testing.R, graphsH0.ipynb, and graphsH1.ipynb
To reproduce **Figures 6** and **7**, run ridgeTrain.R and graphs3.ipynb.

## Section 13.3 of the Supplement:

The codes are stored in `simulations/Step2_Training/EnetFitting_highdim`
To reproduce **Figure 8**, run EnetTrain.R and graphs3.ipynb

## Section 14 of the Supplement:

The codes are stored in `simulations/Step2_Training_Masking_Y`. 
To reproduce **Figure 9**, run runTests.sh, graphsH0.ipynb, and graphsH1.ipynb


## Section 18 of the Supplement:

Check `car_data_extended_experiments/README.txt`

## Section 19.3 of the Supplement:

run `simulations/Step1_Testing_trade_off_experiment/runTests.sh`
run graphsH1.ipynb to generate graphs.

## Section 19.4.1 of the Supplement

run `simulations/Step1_Testing_Urobust_randomU/runTests.sh`
run graphsH0.ipynb and graphsH1.ipynb to generate the plots.

## Section 19.4.2 of the Supplement
run `simulations/Step1_Testing_Urobust_fixedU/runTests_Urobust.sh`  and `simulations/Step1_Testing_Urobust_fixedU/Counts.ipynb`  

