
# Text generation using RNNs on Tensorflow 1.8

This code base illustrates how to use up-to-date Tensorflow APIs (as of May 2018) to train an RNN on a corpus of text
and to use it to generate new, similar-sounding text.

There are already plenty of RNN-based text generators out there, but none that use the Tensorflow high-level Estimator API.

The repository includes a bunch of books by Jules Verne. Training a model on those yields ouput like:


> In a few moments the engineer recognised the dense smoke hidden on the ground, the sharply left with crimson, and the breeze stood behind before numerous pieces of ice so terribly defined a white appearance, which for through the shades in the night the pieces were escaping from this vapory race afforded many [unknown] the articles of many ornaments. at the end of an operation to contain a single [unknown] belonging to new royal, succeeded on beer and [unknown] into idleness; besides, there were strangers and farmers. These indigenous people must be counted by Collinson and [unknown] - the most suitable for many of the various inhabitants of the island, when in the first place, representatives of the Gallian night - Dacosta and Day whose appetite were far less difficult to please? 
>
> "If you do not make," answered Harry, "can you afford any resistance to you?"
>
> He ran back to the trees and inundated the snow. Half-a-dozen cod sounds that was found, however, of harsh stature, each of which was most frequently appreciated to his lofty [unknown]. 
>
> Kalumah. 
>
> "Either I was at my disposal," said I to Lord Edward van Tricasse: 
>
> "Same attraction, Simon," answered Pencroft. "Who knows with those who trust to a very common nature, whether it was the basis of all these misfortunes?"


## Installing

Run

```
  python3 setup.py
```

Note that the code is intended to be used with Python 3.x.

## Training locally

To train the model locally, run something like:

```
python -m trainer.task \
  --data_dir training-data/jules \
  --job-dir job/%JOB% \
  --summary_dir tensorboard-data/%JOB% \
  --data_prefix jules \
  --layers 2 \
  --lstm_state_size 50 \
  --unroll_steps 30 \
  --epochs 3
```

The placeholder `%JOB%` is replaced by the code with a job ID that is composed of the network parameters used for the run, 
e.g. `jules-120-states-4-layers-norm-word` (Jules Verne data set, 120 LSTM states, 4 layers, layer normalization, word-based model).

This makes it easier to keep track of the hyperparameters of each model when you are tuning them. 

Hyperparameters that don't affect the checkpoint format are not part of the job name, so if you e.g. change the learning rate or the number of unrolled steps you can continuing training the same model.

The script will generate some text regularly during training (everytime a checkpoint is saved) to give you some idea of the progress beyond the loss.

## Training on Google Cloud Machine Learning Engine

To train the model on Google Cloud, first following the instructions in the [Tensorflow Getting Started](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#setup) and then run something like:

You need to create "buckets" in Google Cloud Storage where the output data will be stored. There are four kinds of data that need storing.

 * The code to be executed. This is the staging bucket.
 * The model checkpoints generated. These are stored in the job directory.
 * The Tensorboard data that allows you to visualize the training progress.
 * The training data (the books).

In practise, I'd recommend creating two buckets: one for the staging data and one for everything else. The reason is that the staging data cannot be put in a sub-folder and therefore clutters the root.

```
  gsutil mb [STAGING_BUCKET]
  gsutil mb [DATA_BUCKET]
```

Pick any name for the two buckets. Note that bucket names need to be globally unique across all Google Cloud Storage's buckets, so you might need some creativity in the naming.

Then copy the training data to the data bucket using `gsutil`:

```
  gsutil cp -r training-data/* [DATA_BUCKET]/training-data/
```

```
gcloud ml-engine jobs submit training [A_UNIQUE_ID] \
  --package-path trainer \
  --module-name trainer.task \
  --staging-bucket gs://[STAGING_BUCKET] \
  --job-dir gs://[DATA_BUCKET]/job/%JOB% \
  --runtime-version 1.7 \
  --config config.yaml \
  -- \
  --summary_dir gs://[DATA_BUCKET]/tensorboard-data/%JOB% \
  --data_dir gs://[DATA_BUCKET]/training-data \
  --data_prefix jules \
  --epochs=20 \
  --layers=4 \
  --lstm_state_size=120 \
  --unroll_steps=30 \
  --layer_norm=True 
```

## Generating text

To generate text, just replace `task` with `babble-task`. The hyperparameters determine which model (directory) is used, so they also need to be specified.

So to generate text using the model trained with the example command above, the following command could be used. Unnecessary parameters have been dropped, but the exact same parameters as above could also be used.

```
python -m trainer.babble-task \
  --job-dir job/%JOB% \
  --data_prefix jules \
  --layers 2 \
  --lstm_state_size 50 
```

## Hyperparameters

There's plenty of hyperparameters that can be specified as command-line parameter. 
For a complete list see [flags.py](/trainer/flags.py).

Notably, you can specify whether you want to train a character-level model or a word-based model using the parameter `--language_model=word/char`

## Feedback

If you find the code useful or have feedback, bugs or comments, do drop me a line at [andreas.ehrencrona@velik.it](andreas.ehrencrona@velik.it)