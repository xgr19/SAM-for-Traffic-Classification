At first time, run as follow:
1 python3 Preprocess.py  # this will read dataset from /data, and generate data.map
2 python3 train.py # start training and validating after every epoch
3 python3 Cal_metrics.py # calculate recall, presicion, f1 of the best model on validation (run after step 2 is finished)

After first:
2 python3 train.py
3 python3 Cal_metrics.py