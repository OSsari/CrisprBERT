# CrisprBERT

## Needed Packages:

- python==3.7
- tensorflow==2.5.0
- pandas==1.3.5
- numpy==1.19.5
- scikit-learn==1.0.2
- transformers==4.8.2

## bert.py:

You can run **bert.py** in your environment with the appropriate arguments. You can either run a train/validation or a K-cross validation configuration.
Look at the [Parameters](##Parameters) section for more information.

For example, you can run the following command in your local environment:
```
python bert.py --evaluate False --is_k_cross False --training False --file_path example.csv --model_path ./model/model 
```

For more information about the BERT layer parameters, please consult the [transformers](https://huggingface.co/docs/transformers/model_doc/bert) library website.


## Parameters:

- **--evaluate** (`bool`, *required*) - Wether to use the model in prediction mode only. Make sure to have saved model weights with correct model parameters entered. 

- **--is_k_cross** (`bool`, *required*) - In **bert.py** only. If True, K-cross validation training will be run, else, a single train/validation 
training will be run.

- **--training** (`bool`, *required*) - Used in BERT configuration. Whether or not to use the model in training mode 
(some modules like dropout modules have different behaviors between training and evaluation).

- **--file_path** (`str`, *required*) - File path of the data for training or evaluation.

- **--model_path** (`str`, *required*) - File path of the model weights, either to save them or to load them. If you are loading a model, make sure that model parameters match with the saved version.

- **--hidden_size** (`int`, *optional*, defaults to 64) - Dimensionality of the encoder layers and the pooler layer. 

- **--intermediate_size** (`int`, *optional*, defaults to 2048) - Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.

- **--num_attention_heads** (`int`, *optional*, defaults to 8) - Number of attention heads for each attention layer in the Transformer encoder.

- **--num_hidden_layers** (`int`, *optional*, defaults to 6) - Number of hidden layers in the Transformer encoder.

- **--hidden_dropout_prob** (`float`, *optional*, defaults to 0.1) - The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

- **--attention_probs_dropout_prob** (`float`, *optional*, defaults to 0.1) - The dropout ratio for the attention probabilities. 

- **--hidden_act** (`str`, *optional*, defaults to `"gelu"`) - The non-linear activation function (string) in the encoder and pooler. Either "gelu", "relu", "silu" or "gelu_new".

- **--encoding** (`str`, *optional*, defaults to 'doublet') - Encoding type, either 'single', 'doublet' or 'triplet'.

- **--valid_size** (`float`, *optional*, defaults to 0.1) - Size of the validation set for the train/valid/test split. Between 0 (excluding) and 1.

- **--test_size** (`float`, *optional*, defaults to 0.05) - Size of the test set for the train/valid/test split. Between 0 (including) and 1.

- **--n_split** (`int`, *optional*, defaults to 5) - Number of splits for K-cross validation. Has to be minimum 2. 

- **--num_epochs** (`int`, *optional*, defaults to 400) - Number of training epochs.

- **--batch_size** (`int`, *optional*, defaults to 128) - Batch size for training. 

- **--learning_rate** (`float`, *optional*, defaults to 2e-5) - Learning rate for training.

- **--learning_decay** (`str`, *optional*, defaults to false) - Learning rate scheduler  for training. Uses the `ReduceLROnPlateau` callback provided by TensorFlow. 
