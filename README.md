# OICycle: Observe, Infer, and Intervene Cycle
Code repository for release of OICycle: Observe, Infer, and Intervene Cycle for Belief-State Learning in 
Longitudinal Clinical Decision Making


## Data Preprocessing pipeline
1. Run ../datasets_creation/cohort_creation.py  
2. Run ../datasets_creation/cohort_measurements_creation.py
3. Run ../datasets_creation/preprocessing.py
4. Run ../datasets_creation/episodic_data_creation.py
5. Run ../Datasets/dataset_split.py
6. Run ../Datasets/statistical_summarizer.py
7. Run ../Loss/positive_weights.py


## Training
1. Open ../TrainEval/config.py and change the necessary parameters.
2. Run ../TrainEval/main_run.py

## Testing
1. Open ../Metrics/test_config.py and change the necessary parameters.
2. Run ../Metrics/test.py