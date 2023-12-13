NIMRF
=========================


Predicting patient mortality risk facilitates early intervention in intensive care unit (ICU) patients at greater risk of disease progression. Collecting 33798 patients from the MIMIC-III database, we developed an integrated network NIMRF (Network Integrating Memory Module and Random Forest) based on the memory module and random forest module to dynamically predict mortality risk in ICU patients. The above-mentioned two sub-modules respectively use LSTM and random forest as the backbone networks.


## Citation
Please be sure to cite the original [MIMIC-III paper](http://www.nature.com/articles/sdata201635).


## Requirements
We do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. Specifically, download the CSVs. Otherwise, generally we make liberal use of the following packages:

- numpy
- pandas

LSTM models use [Keras](https://keras.io/).


## Creating the model datasets
Here are the required steps to create the model datasets. It assumes that you already have MIMIC-III dataset (lots of CSV files) on the disk.

1. Clone the repo.

       git clone https://github.com/yyy/NIMRF/（工程路径）
       cd NIMRF/
    
2. Take MIMIC-III CSVs, generate one directory per `SUBJECT_ID` and write ICU stay information to `data/{SUBJECT_ID}/stays.csv`, diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`, and events to `data/{SUBJECT_ID}/events.csv`.

       python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/

3. Attempt to fix some issues (ICU stay ID is missing) and remove the events that have missing information. 

       python -m mimic3benchmark.scripts.validate_events data/root/

4. Break up per-subject data into separate episodes (pertaining to ICU stays). 

       python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/

5. Split the whole dataset into training and testing sets.

       python -m mimic3benchmark.scripts.split_train_and_test data/root/
	
6. Generate the dataset which can later be used in NIMRF. These commands are independent, if you are going to work only on one benchmark task, you can run only the corresponding command.

       Generate the dataset that can be used in the NIMRF model to predict a patient's risk of death within 1 hour:
		python -m mimic3benchmark.scripts.create_mortality_somehours data/root/ data/mortality/1h/ 1
		
	   Generate the dataset that can be used in the NIMRF model to predict a patient's risk of death within 1~3 hours:
		python -m mimic3benchmark.scripts.create_mortality_somehours data/root/ data/mortality/3h/ 3
		
	   Generate the dataset that can be used in the NIMRF model to predict a patient's risk of death within 3~6 hours:
		python -m mimic3benchmark.scripts.create_mortality_somehours data/root/ data/mortality/6h/ 6
		
	   Generate the dataset that can be used in the NIMRF model to predict a patient's risk of death within 6~12 hours:
		python -m mimic3benchmark.scripts.create_mortality_somehours data/root/ data/mortality/12h/ 12


## Train / validation splitting
Extract validation set from the training set. This step is required for running the models.

       python -m mimic3models.split_train_val {dataset-directory}
       
`{dataset-directory}` can be either `data/mortality/1h`, `data/mortality/3h`, `data/mortality/6h` or `data/mortality/12h`.


## Training and testing
Memory module training.
       python -um mimic3models.mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/mortality
	   
Memory module testing.
	   python -um mimic3models.mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 1 --normalizer_state mimic3models/mortality/ihm_ts1.0.input_str_previous.start_time_zero.normalizer --load_state {model-directory} --data {test dataset-directory}

Random forest module training.
	   python -um mimic3models.mortality.train_mortality

Random forest module testing and Interpretation.
	   python -um mimic3models.mortality.run_mortality


## Model performance evaluation(AUC)
Evaluate model performance (AUC, 95% confidence interval, etc.)
	   python -um mimic3benchmark.evaluation.evaluateAUC