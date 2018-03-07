# # baseline
python -m estimator_trainer.task --data_dir=../normalization/data/ \
  --data_prefix=jobs_10k_char_level --epochs=6  \
  --layers=2 \
  --batch_size=200 --lstm_state_size=50 --unroll_steps=10 \
  --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 \
  --optimizer=Adam --clip_gradients=5


python -m estimator_trainer.task --data_dir=../normalization/data/ \
  --data_prefix=jobs_10k_char_level --epochs=99 --layers=2 \
  --batch_size=20 --lstm_state_size=650 --unroll_steps=35 \
  --learning_rate=0.02 --lr_decay_rate=0.8 --lr_decay_steps=10000 \
  --optimizer=Adam --clip_gradients=5 --dropout=0.5 --max_steps=2000

# large network that is screwed up despite having fairly low loss
python -m estimator_trainer.task --data_dir=../normalization/data/ --data_prefix=jobs_10k_char_level --epochs=6    --layers=2 --max_steps=500   --batch_size=200 --lstm_state_size=500 --unroll_steps=10   --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --clip_gradients=.5 --layer_norm=true --dropout=0.4

# GRU baseline

python -m estimator_trainer.task --data_dir=../normalization/data/   --data_prefix=jobs_10k_char_level --epochs=6    --layers=2   --batch_size=200 --lstm_state_size=75 --unroll_steps=10   \
  --learning_rate=0.01 --lr_decay_rate=0.9 --lr_decay_steps=10000   --optimizer=Adam --clip_gradients=5


# more everything
python -m estimator_trainer.task --data_dir=../normalization/data/ \
  --data_prefix=jobs_290k_char_level --epochs=6  \
  --layers=2 --max_steps=50000 \
  --batch_size=200 --lstm_state_size=400 --unroll_steps=10 \
  --learning_rate=1.0 --lr_decay_rate=0.7 --lr_decay_steps=10000 --clip_gradients=5

# # no embed
# python -m estimator_trainer.task --data_dir=../normalization/data/ \
#   --data_prefix=jobs_10k_char_level --epochs=6 \
#   --layers=2 \
#   --batch_size=200 --lstm_state_size=50 --unroll_steps=20 \
#   --learning_rate=1.0 --lr_decay_rate=0.5 --lr_decay_steps=10000 --clip_gradients=5

# # one layer
# python -m estimator_trainer.task --data_dir=../normalization/data/ \
#   --data_prefix=jobs_10k_char_level --epochs=6 --embedding_size=10 \
#   --layers=1 \
#   --batch_size=200 --lstm_state_size=50 --unroll_steps=20 \
#   --learning_rate=1.0 --lr_decay_rate=0.5 --lr_decay_steps=10000 --clip_gradients=5

# # double state
# python -m estimator_trainer.task --data_dir=../normalization/data/ \
#   --data_prefix=jobs_10k_char_level --epochs=6 --embedding_size=10 \
#   --layers=2 \
#   --batch_size=200 --lstm_state_size=100 --unroll_steps=20 \
#   --learning_rate=1.0 --lr_decay_rate=0.5 --lr_decay_steps=10000 --clip_gradients=5

# three layers
python -m estimator_trainer.task --data_dir=../normalization/data/ \
  --data_prefix=jobs_10k_char_level --epochs=6  \
  --layers=3 \
  --batch_size=200 --lstm_state_size=50 --unroll_steps=20 \
  --learning_rate=1.0 --lr_decay_rate=0.5 --lr_decay_steps=10000 --clip_gradients=5

# dropout
python -m estimator_trainer.task --data_dir=../normalization/data/ \
  --data_prefix=jobs_10k_char_level --epochs=6  \
  --layers=2 --keep_probs=0.5 \
  --batch_size=200 --lstm_state_size=50 --unroll_steps=20 \
  --learning_rate=1.0 --lr_decay_rate=0.5 --lr_decay_steps=10000 --clip_gradients=5

# halve steps
python -m estimator_trainer.task --data_dir=../normalization/data/ \
  --data_prefix=jobs_10k_char_level --epochs=6 \
  --layers=2 \
  --batch_size=200 --lstm_state_size=50 --unroll_steps=10 \
  --learning_rate=1.0 --lr_decay_rate=0.5 --lr_decay_steps=10000 --clip_gradients=5

# halve embed
python -m estimator_trainer.task --data_dir=../normalization/data/ \
  --data_prefix=jobs_10k_char_level --epochs=6 \
  --layers=2 \
  --batch_size=200 --lstm_state_size=50 --unroll_steps=20 \
  --learning_rate=1.0 --lr_decay_rate=0.5 --lr_decay_steps=10000 --clip_gradients=5

# three layers, quadruple state
python -m estimator_trainer.task --data_dir=../normalization/data/ \
  --data_prefix=jobs_10k_char_level --epochs=6 \
  --layers=3 \
  --batch_size=200 --lstm_state_size=200 --unroll_steps=20 \
  --learning_rate=1.0 --lr_decay_rate=0.5 --lr_decay_steps=10000 --clip_gradients=5



# one of the best locally
python -m estimator_trainer.task 
  --data_dir=../normalization/data/ --data_prefix=jobs_290k_char_level \
  --layers=3 \
  --lstm_state_size=120 \
  --unroll_steps=10 \
  --layer_norm=True \
  --max_steps=20000 \
  --clip_gradients=2 --learning_rate=1.0 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 
  

# big model trained for a couple of hours on gcloud
python -m estimator_trainer.task \
  --data_dir=../normalization/data/ --data_prefix=jobs_290k_char_level.ascii \
  --layers=4 \
  --lstm_state_size=120 \
  --unroll_steps=30 \
  --layer_norm=True \
  --max_steps=75000 \
  --clip_gradients=2 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --batch_size=200 \

# first test with word model
python -m estimator_trainer.task --data_dir=../normalization/data/ \
  --data_prefix=jobs_290k \
  --batch_size=200 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --optimizer=Adam --clip_gradients=2 \
  --language_model=word --vocab_size=500 \
  --layers=2 --lstm_state_size=50 --unroll_steps=10 


# check cloud model
python -m estimator_trainer.babble-task --data_dir=../normalization/data/ \
  --data_prefix=jobs_290k \
  --batch_size=200 --learning_rate=0.02 --lr_decay_rate=0.5 --lr_decay_steps=10000 --optimizer=Adam --clip_gradients=2 \
  \
  --language_model=word \
  --layers=4 \
  --lstm_state_size=120 \
  --unroll_steps=30 \
  --layer_norm=True \
  --max_steps=5000 

