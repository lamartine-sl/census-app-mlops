# Model Card

## Model Details
Lamartine Santana created the model. It is a Catboost Classifier version:0.26.1 using the hyperparameter available in params.yaml
## Intended Use
This model should be used to predict the salary range of a worker american based on demographic data of census.

Your use is intended for academic and study purposes.
## Training Data
The data source:[Census Bureau data obtained from Udacity](https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv).

The dataset was receive a cleaning for reading, available in `census_clean.csv` file.

The data has approximately +30.000 row, and was split in 80% for train and 20% for test, using the `train_test_split` method of sklearn library.

Categorical features were set as:
- `workclass`
- `education`
- `marital_status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native_country`

The remaining columns were set as continuous features.

## Evaluation Data
The data for evaluation is used on the 20% of the data.
## Metrics
THe metrics using in this model was precision (78%), recall(67%) and fbeta score(72%). the metrics was available in `metrics_model.json`
## Ethical Considerations
Census data is publicly available, contained sensitive information about the sex, education, gender and race os people. The output model should not be considered for non-didactic uses, as other important model information was not considered and there was no bias test.
## Caveats and Recommendations
This project was not focused on the search for the best performance of the model, therefore, no deep research was done on hyperparameters or feature engineering. In case the work continues, the suggestion would be to seek more up-to-date and widely available data.