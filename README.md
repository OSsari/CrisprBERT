# CrisprBERT

## Needed Packages:

- python==3.7
- tensorflow==2.5.0
- pandas==1.3.5
- numpy==1.19.5
- scikit-learn==1.0.2
- transformers==4.8.2

## Running CrisprBERT:

You can run **crisprbert.py** in your local environment with the appropriate arguments. To train CrisprBERT on a
dataset, you can either use a single split train/validation scheme or a k-cross validation configuration. Note that a new
model will not be saved while training the model with k-cross validation. This is only used for hyperparameter selection
purposes.
Look at the [Parameters](##Parameters) section for more information.

For example, you can run the following command in your local environment:

```
python crisprbert.py --evaluate False --is_k_cross False --file_path data/train.csv --model_path ./model/model 
```

This command runs CirsprBERT in a training setting where it will leave a single validation set, with the data file train.csv and model path ./model/model. You can then run the following command to evaluate the saved model on a different dataset, say evaluate.csv:

```
python crisprbert.py --evaluate True --file_path data/evaluate.csv --model_path ./model/model 
```

The model will run as long as the model paramters inputted in the evaluate command are the same as those inputted for the training command. Model paramteres can be further customized by using the appropriate commands.  

For more information about the BERT layer parameters, please consult
the [Transformers](https://huggingface.co/docs/transformers/model_doc/bert) library website.

## Data Format:

The format of the datafile must be the same as the Change_seq.csv file. In particular, three columns are necessary: (1) sequence, (2) Target sequence and (3) class. The sequence column is reserved for the sgRNAs, the Target sequence for the off-target sites, and the class for the binary classifications (1 for positive off-targets, 0 otherwise). Note that no indel sequences are allowed. The NGG sequence must be present (hence 22 nucleotide sequences) and only the four nucleatide letters are recognized (A, C, G, T). Furthermore, the datafile must be placed in a folder called `data`. 

## Parameters:

- **--evaluate** (`bool`, *required*) - Whether to use the model in prediction mode. Make sure that the saved model
  parameters match with the parameters entered as arguments.

- **--is_k_cross** (`bool`, *required if --evaluate is False*) - If True, K-cross validation training will be run. If
  False, a single split train/validation training will be run. Note that a new
model will not be saved while training the model with k-cross validation.

[//]: # (- **--training** &#40;`bool`, *required*&#41; - Used in BERT configuration. Whether to use the model in training mode. Some)
[//]: # (  modules like dropout modules have different behaviors between training and evaluation.)

- **--file_path** (`str`, *required*) - File path of the data for training or evaluation.

- **--model_path** (`str`, *required*) - File path of the model weights, either to save them or to load them. If you are
  loading a model, make sure that model parameters entered as arguments match with the saved version.

- **--hidden_size** (`int`, *optional*, defaults to 64) - Dimensionality of the encoder layers and the pooler layer.

- **--intermediate_size** (`int`, *optional*, defaults to 2048) - Dimensionality of the “intermediate” (often named
  feed-forward) layer in the Transformer encoder.

- **--num_attention_heads** (`int`, *optional*, defaults to 8) - Number of attention heads for each attention layer in
  the Transformer encoder.

- **--num_hidden_layers** (`int`, *optional*, defaults to 6) - Number of hidden layers in the Transformer encoder.

- **--hidden_dropout_prob** (`float`, *optional*, defaults to 0.1) - The dropout probability for all fully connected
  layers in the embeddings, encoder, and pooler.

- **--attention_probs_dropout_prob** (`float`, *optional*, defaults to 0.1) - The dropout ratio for the attention
  probabilities.

- **--hidden_act** (`str`, *optional*, defaults to `"gelu"`) - The non-linear activation function (string) in the
  encoder and pooler. Either "gelu", "relu", "silu" or "gelu_new".

- **--encoding** (`str`, *optional*, defaults to 'doublet') - Encoding type, either 'single', 'doublet' or 'triplet'.

- **--valid_size** (`float`, *optional*, defaults to 0.1) - Size of the validation set for the train/valid/test split.
  Between 0 (excluding) and 1.

- **--test_size** (`float`, *optional*, defaults to 0.05) - Size of the test set for the train/valid/test split. Between
  0 (including) and 1.

- **--n_split** (`int`, *optional*, defaults to 5) - Number of splits for K-cross validation. Has to be minimum 2.

- **--num_epochs** (`int`, *optional*, defaults to 400) - Number of training epochs.

- **--batch_size** (`int`, *optional*, defaults to 128) - Batch size for training.

- **--learning_rate** (`float`, *optional*, defaults to 2e-5) - Learning rate for training.

- **--learning_decay** (`str`, *optional*, defaults to false) - Learning rate scheduler for training. Uses
  the `ReduceLROnPlateau` callback provided by TensorFlow. 
