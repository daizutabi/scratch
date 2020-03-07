# #!

# # How to train Boosted Trees models in TensorFlow
# # (https://www.tensorflow.org/alpha/tutorials/estimators/boosted_trees)

import altair as alt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve

tf.random.set_seed(123)
# ## Load the titanic dataset
# !Load dataset.
dftrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
dfeval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv")
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")

# ## Explore the data
dftrain.head()
# -
dftrain.describe()
# -
dftrain.shape[0], dfeval.shape[0]
# -
chart = alt.Chart(dftrain).mark_bar()
chart.encode(alt.X("age:Q", bin={"maxbins": 20}), y="count()")
# -
chart.encode(x="count()", y="sex", color="sex")
# -
chart.encode(x="count()", y="class", color="class")
# -
chart.encode(x="count()", y="embark_town", color="embark_town")
# -
df = pd.concat([dftrain, y_train], axis=1)
alt.Chart(df).mark_bar().encode(x="mean(survived)", y="sex", color="sex")

# ## Create feature columns and input functions
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
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab)
    )


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # Need to one-hot encode categorical features.
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(
        tf.feature_column.numeric_column(feature_name, dtype=tf.float32)
    )
# -
example = dict(dftrain.head(1))
class_fc = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        "class", ("First", "Second", "Third")
    )
)
print('Feature value: "{}"'.format(example["class"].iloc[0]))
print("One-hot encoded: ", tf.keras.layers.DenseFeatures([class_fc])(example).numpy())
# -
tf.keras.layers.DenseFeatures(feature_columns)(example).numpy()

# -
# !Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(y_train)


def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset

    return input_fn


# -
# !Training and evaluation input functions.
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

# ## Train and evaluate the model
linear_est = tf.estimator.LinearClassifier(feature_columns)

# !Train model.
linear_est.train(train_input_fn, max_steps=100)

# -
# !Evaluation.
result = linear_est.evaluate(eval_input_fn)
pd.Series(result)

# -
# !Since data fits into memory, use entire dataset per layer.
# !It will be faster. Above one batch is defined as the entire dataset.
n_batches = 1
est = tf.estimator.BoostedTreesClassifier(
    feature_columns, n_batches_per_layer=n_batches
)

# !The model will stop training once the specified number of trees is built,
# !not based on the number of steps.
est.train(train_input_fn, max_steps=100)

# -
# !Eval.
result = est.evaluate(eval_input_fn)
pd.Series(result)
# -
pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred["probabilities"][1] for pred in pred_dicts], name="probability")
alt.Chart(probs.to_frame()).mark_bar().encode(
    alt.X("probability", bin={"maxbins": 20}), y="count()"
)
# -
fpr, tpr, _ = roc_curve(y_eval, probs)
df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
alt.Chart(df).mark_line().encode(
    x=alt.X("fpr", title="false positive rate"),
    y=alt.Y("tpr", title="true positive rate"),
).properties(title="ROC curve")
