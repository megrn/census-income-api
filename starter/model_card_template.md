# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a Random Forest classifier trained with scikit-learn to predict
whether a census record belongs to the `>50K` or `<=50K` income class. Categorical
features are one-hot encoded and the target label is binarized before training.

## Intended Use
This model is intended for the Udacity Census Income API project as a supervised
learning baseline and as an example of packaging a trained model behind a
RESTful API. It is not intended for real employment, lending, housing, or other
high-stakes eligibility decisions.

## Training Data
The training data is the UCI Census Income dataset provided with the project.
The CSV is cleaned by stripping whitespace from column names and categorical
values. The data is split into an 80% training set and a 20% test set using a
fixed random seed and stratification on the salary label.

## Evaluation Data
The evaluation data is the held-out 20% test split that is not used during model
training. Slice performance is also reported for each value of the categorical
features in `slice_output.txt`.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
The model is evaluated using precision, recall, and F1 score (`fbeta` with
`beta=1`). On the held-out test split from the completed run:

* Precision: 0.7327
* Recall: 0.6397
* F1 / F-beta: 0.6830

Slice-level precision, recall, and F1 are written to `slice_output.txt`.

## Ethical Considerations
The census data contains sensitive demographic attributes, including sex, race,
age, and native country. A model trained on these features can learn and
reproduce historical inequities present in the data. Predictions should therefore
be audited across demographic slices and should not be used as a sole basis for
decisions affecting people.

## Caveats and Recommendations
This is a baseline model with limited feature engineering. Before production
use, the model should receive deeper fairness analysis, more robust monitoring,
model drift checks, and review by domain experts. The API should also be deployed
with authentication, rate limiting, logging, and a repeatable model release
process.
