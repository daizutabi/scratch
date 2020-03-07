# # Gradient Boosted Trees: Model understanding
# # (https://www.tensorflow.org/alpha/tutorials/estimators/
# # boosted_trees_model_understanding)
import altair as alt
import numpy as np
import pandas as pd
import tensorflow as tf

from ivory.utils.altair import bar_from_series

tf.random.set_seed(123)
# ## How to interpret Boosted Trees models both locally and globally

# ## Load the titanic dataset
# !Load dataset.
dftrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
dfeval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv")
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")


# ## Create feature columns, input_fn, and the train the estimator
# ### Preprocess the data
fc = tf.feature_column
CATEGORICAL_COLUMNS = [
    "sex",
    "n_siblings_spouses",
    "parch",
    "class",
    "deck",
    "embark_town",
    "alone",
]
NUMERIC_COLUMNS = ["age", "fare"]


def one_hot_cat_column(feature_name, vocab):
    return fc.indicator_column(
        fc.categorical_column_with_vocabulary_list(feature_name, vocab)
    )


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # Need to one-hot encode categorical features.
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(fc.numeric_column(feature_name, dtype=tf.float32))

# ### Build the input pipeline
# !Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(y_train)


def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient="list"), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs).batch(NUM_EXAMPLES)
        return dataset

    return input_fn


# !Training and evaluation input functions.
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

# ### Train the model
params = {
    "n_trees": 50,
    "max_depth": 3,
    "n_batches_per_layer": 1,
    # You must enable center_bias = True to get DFCs. This will force the model to
    # make an initial prediction before using any features (e.g. use the mean of
    # the training labels for regression or log odds for classification when
    # using cross entropy loss).
    "center_bias": True,
}

est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
# !Train model.
est.train(train_input_fn, max_steps=100)

# -
# !Evaluation.
results = est.evaluate(eval_input_fn)
pd.Series(results).to_frame()

# ## Local interpretability
pred_dicts = list(est.experimental_predict_with_explanations(eval_input_fn))

# !Create DFC Pandas dataframe.
labels = y_eval.values
probs = pd.Series([pred["probabilities"][1] for pred in pred_dicts])
df_dfc = pd.DataFrame([pred["dfc"] for pred in pred_dicts])
df_dfc.describe().T

# -
# !Sum of DFCs + bias == probabality.
bias = pred_dicts[0]["bias"]
dfc_prob = df_dfc.sum(axis=1) + bias
np.testing.assert_almost_equal(dfc_prob.values, probs.values)


# Plot results.
ID = 182
example = df_dfc.iloc[ID]  # Choose ith example from evaluation set.
example.name = "dfc"
df = example.to_frame()
df.index.name = "feature"
df.reset_index(inplace=True)
df["abs_dfc"] = df["dfc"].abs()
df.sort_values("abs_dfc", inplace=True)
alt.Chart(df).mark_bar().encode(
    x="dfc",
    y=alt.Y(
        "feature:O",
        sort=alt.EncodingSortField(field="abs_dfc", op="values", order="descending"),
    ),
    color=alt.condition(alt.datum.dfc > 0, alt.value("green"), alt.value("orange")),
)


# ## Global feature importances
# ### Gain-based feature importances
importances = est.experimental_feature_importances(normalize=True)
s_imp = pd.Series(importances)
s_imp
# !Visualize importances.
bar_from_series(s_imp, "gfi", "feature")

# ### Average absolute DFCs
dfc_mean = df_dfc.abs().mean().sort_values()
bar_from_series(dfc_mean, "dfc", "feature")
# -
FEATURE = "fare"
df = pd.concat([df_dfc[[FEATURE]], dfeval[[FEATURE]]], axis=1)
df.columns = ["contribution", "fare"]
alt.Chart(df).mark_circle().encode(x="fare", y="contribution")

# ### Permutation feature importance


def permutation_importances(est, X_eval, y_eval, metric, features):
    """Column by column, shuffle values and observe effect on eval set.

    source: http://explained.ai/rf-importance/index.html
    A similar approach can be done during training. See "Drop-column importance"
    in the above article."""
    baseline = metric(est, X_eval, y_eval)
    imp = []
    for col in features:
        save = X_eval[col].copy()
        X_eval[col] = np.random.permutation(X_eval[col])
        m = metric(est, X_eval, y_eval)
        X_eval[col] = save
        imp.append(baseline - m)
    return np.array(imp)


def accuracy_metric(est, X, y):
    """TensorFlow estimator accuracy."""
    eval_input_fn = make_input_fn(X, y=y, shuffle=False, n_epochs=1)
    return est.evaluate(input_fn=eval_input_fn)["accuracy"]


features = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
importances = permutation_importances(est, dfeval, y_eval, accuracy_metric, features)
df_imp = pd.Series(importances, index=features)
bar_from_series(df_imp.abs().sort_values(), "permutation_importance", "feature")
