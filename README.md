
# AttnBiLSTM-Sentiment-Detection
An attention based method for neural sentiment detection.

### Instructions
-----
To train a new model with the default settings, enter the `src/` from the command line and type:

`python main.py --bidirectional --earlystop --modelname {model name here}`

This will automatically train and evaluate a new model.

To test and existing model without training add the `--test` flag.

`python main.py --bidirectional --earlystop --modelname {model name here} --test`

### Usage
-----
```
usage: main.py [-h] [--batch_size BATCH_SIZE] [--bidirectional] [--clip CLIP]
               [--dropout DROPOUT] [--earlystop] [--embed_size EMBED_SIZE]
               [--epochs EPOCHS] [--hidden_size HIDDEN_SIZE] [--lr LR]
               [--modelname MODELNAME] [--momentum MOMENTUM]
               [--optimizer OPTIMIZER] [--test]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        number of samples per batch, default=32
  --bidirectional       flag to make a BiLSTM, recommended
  --clip CLIP           amount for gradient clipping, default=0.1
  --dropout DROPOUT     fights overfitting with lstm dropout, default=30
  --earlystop           use best model from training epochs, not always last,
                        recommended
  --embed_size EMBED_SIZE
                        sets length of embedding vectors, default=300
  --epochs EPOCHS       number of epochs for training, default=20
  --hidden_size HIDDEN_SIZE
                        number of neurons in the hidden LSTM layer,
                        default=512
  --lr LR               learning rate for training, default=0.0001
  --modelname MODELNAME
                        name of model for saving, default='model.pt'
  --momentum MOMENTUM   momentum hyperparameter, applies only for RMSprop or
                        SGD optimizers, default=0.9
  --optimizer OPTIMIZER
                        optimizer to use for training, default='rmsprop'
  --test                evaluate an existing model, --modelname must be
                        provided
```