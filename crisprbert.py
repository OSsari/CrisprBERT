"""
Main CrisprBert executable file
"""

import csv
import argparse
from utils import *
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import transformers
from sklearn.model_selection import train_test_split


# %% Hyper parameters
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", type=str.lower, choices=["true", "false"], required=True,
                        help="This argument is required. Either true or false.")
    parser.add_argument("--is_k_cross", type=str.lower, choices=["true", "false"],
                        help="This argument is required when --evaluate is False. Either true or false.")
    parser.add_argument("--training", type=str.lower, choices=["true", "false"], required=True,
                        help="This argument is required. Either true or false.")
    parser.add_argument("--file_path", type=str, required=True,
                        help="This argument is required. Please enter your data file path.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="This argument is required. Please enter your model save or load path.")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--hidden_act", type=str, default="gelu")
    parser.add_argument("--encoding", type=str, default='doublet')
    parser.add_argument("--valid_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--learning_decay", type=str.lower, choices=["true", "false"], default="false")
    parser.add_argument("--n_split", type=int, default=5)

    return parser.parse_args()


def encoding_params(encoding):
    vocab_size = 0
    input_shape = 0

    if encoding == 'single':
        vocab_size = 16
        input_shape = 23
    elif encoding == 'doublet':
        vocab_size = 256
        input_shape = 22
    elif encoding == 'triplet':
        vocab_size = 4096
        input_shape = 21

    return vocab_size, input_shape


# %% Metrics

METRICS = [
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
    tf.keras.metrics.AUC(name='roc_auc', curve='ROC'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn')
]


# %% Creating the model
def create_model(vocab_size, input_shape, hidden_size, intermediate_size, hidden_act, num_attention_heads,
                 num_hidden_layers,
                 hidden_dropout_prob, attention_probs_dropout_prob, learning_rate, training):
    config = transformers.BertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=25,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=1,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        num_labels=2
    )

    Bert_layer = transformers.TFBertModel(config)
    input_ids = tf.keras.Input(shape=(input_shape,), name='input_tokens', dtype='int32')
    X = Bert_layer(input_ids, training=training, return_dict=False)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size))(X.last_hidden_state)
    # .last_hidden_state[:, 1:, :] if [CLS] token present

    X = tf.keras.layers.Dense(128, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(64, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(32, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.4)(X)

    # output = tf.keras.layers.Dense(1, activation='sigmoid')(X.last_hidden_state[:, 0, :])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(X)
    model = tf.keras.Model(inputs=[input_ids], outputs=output)

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=METRICS)
    model.summary()
    return model


# %% Data pre-treating and recovery
def get_data(encoding, file_path):
    C = base_pair()
    base_list = C.create_dict(encoding)
    S = off_tar_read(file_path, base_list)
    encode_matrix, class_labels = S.encode(encoding)
    return encode_matrix, class_labels, base_list


# %% Run CrispBERT on a single train/validation/test configuration
def crisp_bert_single_split(encode_matrix, class_labels, vocab_size, input_shape, hidden_size, intermediate_size,
                            hidden_act,
                            num_attention_heads, num_hidden_layers,
                            hidden_dropout_prob, attention_probs_dropout_prob, learning_rate,
                            learning_decay,
                            valid_size, test_size, batch_size, num_epochs, model_pathway, training):
    if test_size != 0:
        [encode_matrix, testing_seq, class_labels, testing_label] = train_test_split(encode_matrix,
                                                                                     class_labels, test_size=test_size,
                                                                                     random_state=25)

    [training_seq, valid_seq, training_label, valid_label] = train_test_split(encode_matrix,
                                                                              class_labels, test_size=valid_size,
                                                                              random_state=30)

    model = create_model(vocab_size, input_shape, hidden_size, intermediate_size, hidden_act, num_attention_heads,
                         num_hidden_layers,
                         hidden_dropout_prob, attention_probs_dropout_prob, learning_rate, training)

    if learning_decay:
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        callbacks = [lr_scheduler, tf.keras.callbacks.CSVLogger("results.csv")]
    else:
        callbacks = [tf.keras.callbacks.CSVLogger("results.csv")]
    model.fit(training_seq, training_label,
              batch_size=batch_size,
              epochs=num_epochs,
              validation_data=(valid_seq, valid_label),
              callbacks=callbacks,
              verbose=2)

    model.save_weights(model_pathway)

    if test_size != 0:
        model = create_model(vocab_size, input_shape, hidden_size, intermediate_size, num_attention_heads,
                             num_hidden_layers,
                             hidden_dropout_prob, attention_probs_dropout_prob, learning_rate, training=False)
        model.load_weights(model_pathway)
        results = model.evaluate(x=testing_seq, y=testing_label)
        print(results)


# %% Run CrispBERT with K-cross validation
def crisp_bert_k_fold(encode_matrix, class_labels, base_list, vocab_size, input_shape, hidden_size, intermediate_size,
                      hidden_act, num_attention_heads,
                      num_hidden_layers,
                      hidden_dropout_prob, attention_probs_dropout_prob, learning_rate,
                      learning_decay,
                      batch_size, num_epochs, encoding, n_split, training):
    from sklearn.model_selection import KFold

    if not os.path.exists('k_fold_cross_validation_results'):
        os.mkdir('k_fold_cross_validation_results')

    number = 1
    with open('./k_cross_validation_results.csv', 'w', newline='') as result:
        wtr = csv.writer(result)
        wtr.writerow(('off_target', 'on_target', 'labels', 'CrispBERT score'))

        for train_index, test_index in KFold(n_split, shuffle=True, random_state=30).split(encode_matrix, class_labels):
            training_seq, valid_seq = encode_matrix[train_index], encode_matrix[test_index]
            training_label, valid_label = class_labels[train_index], class_labels[test_index]

            model = create_model(vocab_size, input_shape, hidden_size, intermediate_size, hidden_act,
                                 num_attention_heads,
                                 num_hidden_layers,
                                 hidden_dropout_prob, attention_probs_dropout_prob, learning_rate, training)
            name = './k_fold_cross_validation_results/results_fold_' + str(number) + '.csv'
            print('----------------------------------------------------------------------------------')
            print(f'Training for fold {number} ...')

            if learning_decay:
                lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
                callbacks = [lr_scheduler, tf.keras.callbacks.CSVLogger(name)]
            else:
                callbacks = [tf.keras.callbacks.CSVLogger(name)]

            model.fit(training_seq, training_label,
                      batch_size=batch_size,
                      epochs=num_epochs,
                      validation_data=(valid_seq, valid_label),
                      callbacks=callbacks,
                      verbose=2)

            output = model.predict(valid_seq)
            L = GetSeq(valid_seq, base_list)
            on_tar, off_tar = L.get_reverse(encoding)

            for i in range(1, len(output)):
                wtr.writerow((off_tar[i], on_tar[i], valid_label[i], output[i][0]))

            number += 1
            tf.keras.backend.clear_session()


# %% Model evaluate
def crisp_bert(encode_matrix, class_labels, base_list, vocab_size, input_shape, hidden_size, intermediate_size,
               hidden_act, num_attention_heads,
               num_hidden_layers,
               hidden_dropout_prob, attention_probs_dropout_prob, learning_rate,
               encoding, model_pathway):
    model = create_model(vocab_size, input_shape, hidden_size, intermediate_size, hidden_act,
                         num_attention_heads,
                         num_hidden_layers,
                         hidden_dropout_prob, attention_probs_dropout_prob, learning_rate, training=False)
    model.load_weights(model_pathway)
    results = model.predict(x=encode_matrix)

    with open('./crispBERT_results.csv', 'w', newline='') as result:
        wtr = csv.writer(result)
        wtr.writerow(('off_target', 'on_target', 'labels', 'CrispBERT score'))

        L = GetSeq(encode_matrix, base_list)
        on_tar, off_tar = L.get_reverse(encoding)
        for i in range(len(results)):
            wtr.writerow((off_tar[i], on_tar[i], class_labels[i], results[i][0]))


# %% main

if __name__ == "__main__":
    args = parse_arguments()
    args.evaluate = args.evaluate == "true"
    if not args.evaluate and args.is_k_cross is None:
        argparse.ArgumentParser().error("--is_k_cross is required when --evaluate is False")
    args.is_k_cross = args.is_k_cross == "true"
    args.training = args.training == "true"
    args.learning_decay = args.learning_decay == "true"

    vocab_size, input_shape = encoding_params(args.encoding)
    encode_matrix, class_labels, base_list = get_data(args.encoding, args.file_path)

    if args.evaluate:
        crisp_bert(encode_matrix, class_labels, base_list, vocab_size, input_shape, args.hidden_size,
                   args.intermediate_size, args.hidden_act, args.num_attention_heads,
                   args.num_hidden_layers,
                   args.hidden_dropout_prob, args.attention_probs_dropout_prob,
                   args.learning_rate,
                   args.encoding, args.model_path)
    else:
        if args.is_k_cross:
            crisp_bert_k_fold(encode_matrix, class_labels, base_list, vocab_size, input_shape, args.hidden_size,
                              args.intermediate_size, args.hidden_act, args.num_attention_heads,
                              args.num_hidden_layers,
                              args.hidden_dropout_prob, args.attention_probs_dropout_prob,
                              args.learning_rate,
                              args.learning_decay,
                              args.batch_size, args.num_epochs, args.encoding, args.n_split, args.training)
        else:
            crisp_bert_single_split(encode_matrix, class_labels, vocab_size, input_shape, args.hidden_size,
                                    args.intermediate_size, args.hidden_act, args.num_attention_heads,
                                    args.num_hidden_layers,
                                    args.hidden_dropout_prob, args.attention_probs_dropout_prob,
                                    args.learning_rate,
                                    args.learning_decay,
                                    args.valid_size, args.test_size, args.batch_size, args.num_epochs,
                                    args.model_path,
                                    args.training)
