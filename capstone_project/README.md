# Welcome to Zillow project!

## Software requerimentes:

- Python3
- tensorflow
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- tqdm
- xgboost

## Preprocessing the data:

**To download the datasets you have to go to:**
https://www.kaggle.com/c/zillow-prize-1/data

And make sure to download these files.

* properties_2016.csv.zip
* train_2016_v2.csv.zip
* zillow_data_dictionary.xlsx.zip


And extract all files into the folder data/

Run:
`export PYTHONPATH="${PYTHONPATH}:/my/path/to/zillow_project/"`

Then run:
`python3 zillow_project/data.py`

## Baseline
And for running the baseline model:
`python3 zillow_project/baseline.py`

## Classification and Regression tasks
And finally for running classification and regression using LabelEncoding:
`python3 zillow_project/experiment.py`


It is also possible to change parameters and datasets as you want in these files:
~~~~
zillow_project/
	mlp_model.py
	classification_model.py

~~~~
Thanks for reading this! :)


