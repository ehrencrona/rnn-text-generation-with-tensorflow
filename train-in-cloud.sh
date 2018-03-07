#!/bin/bash

gcloud ml-engine jobs submit training onthejob_char_level_`date '+%Y%m%d_%H%M%S'` \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_10k_char_level \
  --epochs=99 --layers=2 \
  --batch_size=20 --lstm_state_size=650 --unroll_steps=35 \
  --learning_rate=0.02 --lr_decay_rate=0.8 --lr_decay_steps=10000 \
  --optimizer=Adam --clip_gradients=5 --dropout=0.5 --max_steps=2000


## attempt to set same parameters as PTB

gcloud ml-engine jobs submit training ptb_large_model_adam \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_10k_char_level \
  --epochs=99 --layers=2 \
  --batch_size=20 --lstm_state_size=650 --unroll_steps=35 \
  --learning_rate=0.02 --lr_decay_rate=0.8 --lr_decay_steps=10000 \
  --optimizer=Adam --clip_gradients=5 --dropout=0.5 --max_steps=2000

gcloud ml-engine jobs submit training ptb_large_model_sgd_x \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_10k_char_level \
  --epochs=99 --layers=2 \
  --batch_size=20 --lstm_state_size=650 --unroll_steps=35 \
  --learning_rate=0.02 --lr_decay_rate=0.8 --lr_decay_steps=10000 \
  --optimizer=SGD --clip_gradients=5 --dropout=0.5 --max_steps=2000

# uncompressed input files with suffix .ascii
gcloud ml-engine jobs submit training ptb_large_model_sgd_ascii_uncomp \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_10k_char_level.ascii \
  --epochs=99 --layers=2 \
  --batch_size=20 --lstm_state_size=650 --unroll_steps=35 \
  --learning_rate=0.02 --lr_decay_rate=0.8 --lr_decay_steps=10000 \
  --optimizer=SGD --clip_gradients=5 --dropout=0.5 --max_steps=2000

# is the issue fixed? (290k asciiencoded)
gcloud ml-engine jobs submit training retest_290k \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k_char_level.ascii \
  --epochs=99 --layers=2 \
  --batch_size=20 --lstm_state_size=650 --unroll_steps=35 \
  --learning_rate=0.02 --lr_decay_rate=0.8 --lr_decay_steps=10000 \
  --optimizer=SGD --clip_gradients=5 --dropout=0.5 --max_steps=2000

# one of the best locally. how does it do in the long run?
# sub 1.0 loss after 8000 steps with only 10 unroll
gcloud ml-engine jobs submit training layers3state120unroll10layernormlr10e_2 \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k_char_level.ascii \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=20000 --batch_size=200 \
  \
  --layers=3 \
  --lstm_state_size=120 \
  --unroll_steps=10 \
  --layer_norm=True \
  --max_steps=20000 

# does it scale to 4 layers?
# not sure, it's slow and still converging
gcloud ml-engine jobs submit training layers4state120unroll10layernormlr10e_2b \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k_char_level.ascii \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \
  \
  --layers=4 \
  --lstm_state_size=120 \
  --unroll_steps=10 \
  --layer_norm=True \
  --max_steps=25000 

# can we handle 20 steps?
# yes
gcloud ml-engine jobs submit training layers3state120unroll20layernormlr10e_2 \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k_char_level.ascii \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \
  \
  --layers=3 \
  --lstm_state_size=120 \
  --unroll_steps=20 \
  --layer_norm=True \
  --max_steps=2000 

# does well locally, but does it continue improving?
gcloud ml-engine jobs submit training layers2state160unroll10lr10e_2 \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k_char_level.ascii \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \
  \
  --layers=2 \
  --lstm_state_size=160 \
  --unroll_steps=10 \
  --max_steps=8000

# does it handle 20 unroll?
# yes, first to achieve < 1.0 loss!
gcloud ml-engine jobs submit training layers2state160unroll20lr10e_2 \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k_char_level.ascii \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \
  \
  --layers=2 \
  --lstm_state_size=160 \
  --unroll_steps=20 \
  --max_steps=8000

# does it handle 40 unroll?
# it does
gcloud ml-engine jobs submit training layers2state160unroll40lr10e_2 \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k_char_level.ascii \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \
  \
  --layers=2 \
  --lstm_state_size=160 \
  --unroll_steps=40 \
  --max_steps=8000

# does it handle 80 unroll?
gcloud ml-engine jobs submit training layers2state160unroll80lr10e_2 \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k_char_level.ascii \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \
  \
  --layers=2 \
  --lstm_state_size=160 \
  --unroll_steps=80 \
  --max_steps=20000

# try again to reproduce the ptb results
# fails. loss 3
gcloud ml-engine jobs submit training ptb_large_model_sgd_again3 \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k_char_level.ascii \
  --epochs=99 --layers=2 \
  --batch_size=20 --lstm_state_size=650 --unroll_steps=35 \
  --learning_rate=0.02 --lr_decay_rate=0.8 --lr_decay_steps=10000 \
  --optimizer=SGD --clip_gradients=5 --dropout=0.5 --max_steps=2000


# 4 layers 40 unroll doesnt work
gcloud ml-engine jobs submit training layers4state120unroll40layernormlr10e_2b \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k_char_level.ascii \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \
  \
  --layers=4 \
  --lstm_state_size=120 \
  --unroll_steps=40 \
  --layer_norm=True \
  --max_steps=75000 

# trying 3 layers 40 unroll insteadl
gcloud ml-engine jobs submit training layers3state120unroll40layernormlr10e_2c \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k_char_level.ascii \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \
  \
  --layers=3 \
  --lstm_state_size=120 \
  --unroll_steps=40 \
  --layer_norm=True \
  --max_steps=75000 

# what about 4 layers 30 unroll?
# winner
gcloud ml-engine jobs submit training layers4state120unroll30layernormlr10e_2 \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k_char_level.ascii \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \
  \
  --layers=4 \
  --lstm_state_size=120 \
  --unroll_steps=30 \
  --layer_norm=True \
  --max_steps=75000 

# char-level winner model training on words
gcloud ml-engine jobs submit training wordslayers4state120unroll30layernormlr10e_2c \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \
  \
  --language_model=word \
  --layers=4 \
  --lstm_state_size=120 \
  --unroll_steps=30 \
  --layer_norm=True \
  --max_steps=5000 

# simple word model
# fails to converge
gcloud ml-engine jobs submit training wordslayers2state120unroll30layerlr10e_2c \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \
  \
  --language_model=word \
  --layers=2 \
  --lstm_state_size=120 \
  --unroll_steps=30 \
  --max_steps=20000

# trying to increase state size
# fails to converge
gcloud ml-engine jobs submit training wordslayers2state240unroll30layerlr10e_2c \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \
  \
  --language_model=word \
  --layers=2 \
  --lstm_state_size=240 \
  --unroll_steps=30 \
  --max_steps=20000

# adding a projection level to char-level winner model
gcloud ml-engine jobs submit training wordslayers4state120unroll30layernorm_proj_lr10e_2c \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/char-level/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \
  \
  --language_model=word \
  --layers=4 \
  --lstm_state_size=120 \
  --unroll_steps=30 \
  --layer_norm=True \
  --num_proj=5000 \
  --max_steps=5000 

# renamed folders on gs
# increased steps 
# -> loss stagnated at around 3.5
gcloud ml-engine jobs submit training jobs_290k_120_states_4_layers_norm_word
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/word/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jobs_290k \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \
  \
  --language_model=word \
  --layers=4 \
  --lstm_state_size=120 \
  --unroll_steps=40 \
  --layer_norm=True \
  --max_steps=10000 

# jules verne, regenerated data without gutenberg notices, 20 epochs, no max steps (should be about 10,000 steps) 
gcloud ml-engine jobs submit training jules_120_states_4_layers_norm_word_2_epochsd \
  --package-path estimator_trainer \
  --module-name estimator_trainer.task \
  --staging-bucket gs://onthejob-training \
  --job-dir gs://onthejob-training/trained-model/jules/%JOB% \
  --runtime-version 1.6 \
  --config config.yaml \
  -- \
  --summary_dir gs://onthejob-training/tensorboard-data/%JOB% \
  --data_dir gs://onthejob-training/training-data \
  --data_prefix jules \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=5000 --batch_size=200 \
  \
  --epochs=20 \
  --language_model=word \
  --layers=4 \
  --lstm_state_size=120 \
  --unroll_steps=30 \
  --layer_norm=True 

  