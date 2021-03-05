from distutils.version import LooseVersion
# from tensorflow.contrib import seq2seq
from sklearn.metrics import f1_score
from tqdm import tqdm_notebook

import numpy as np
import os
import tensorflow as tf
import warnings
import pickle
import gc
import matplotlib.pyplot as plt

from py_util import net_batches
from py_util.preprocessing import PreProcess
from py_util.MultiHeadAttention import MultiHeadAttention



# Precisa arrumar o build rnn quando tem mais de uma camada recorrente, sao retornados os outputs de cada camada recorrente
# Precisa arrumar o build rnn quando tem mais de uma camada recorrente, sao retornados os outputs de cada camada recorrente
# Precisa arrumar o build rnn quando tem mais de uma camada recorrente, sao retornados os outputs de cada camada recorrente
# Precisa arrumar o build rnn quando tem mais de uma camada recorrente, sao retornados os outputs de cada camada recorrente
# Precisa arrumar o build rnn quando tem mais de uma camada recorrente, sao retornados os outputs de cada camada recorrente
# Precisa arrumar o build rnn quando tem mais de uma camada recorrente, sao retornados os outputs de cada camada recorrente
# Precisa arrumar o build rnn quando tem mais de uma camada recorrente, sao retornados os outputs de cada camada recorrente
# Precisa arrumar o build rnn quando tem mais de uma camada recorrente, sao retornados os outputs de cada camada recorrente
# Precisa arrumar o build rnn quando tem mais de uma camada recorrente, sao retornados os outputs de cada camada recorrente
# Precisa arrumar o build rnn quando tem mais de uma camada recorrente, sao retornados os outputs de cada camada recorrente
# Precisa arrumar o build rnn quando tem mais de uma camada recorrente, sao retornados os outputs de cada camada recorrente
class LSTM_Network():
    def __init__(self, pre_proc, attention = True, attn_depth_model=512, attn_num_heads=8, bi_dir=True, dropout_prob=0.5, embed_dim = 256, n_rec_layers = 2, lstm_layer_size = 128,
                 f1_report=None, load_params=False, out_dim=None, save_dir=None, save_params=True, verbose=False):
        """:param save_dir: Directory to save the checkpoints"""
        
        # Check TensorFlow Version
        assert LooseVersion(tf.__version__) >= LooseVersion('1.3'), 'Please use TensorFlow version 1.3 or newer'
        if verbose: print('TensorFlow Version: {}'.format(tf.__version__))

        # Check for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found. Please use a GPU to train your neural network.')
        else:
            if verbose: print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

        self.pre_proc = pre_proc
        if load_params:
            params = pickle.load(open(f"{save_dir}/lstm_net.pkl", "rb"))
            (self.dropout_prob, self.embed_dim, self.n_rec_layers, self.lstm_layer_size,
             self.f1_report, self.out_dim, self.save_dir,self.depth_attn_model, self.num_attn_heads) = params
            self.bi_dir = True
            self.attention = True
            # self.depth_attn_model = attn_depth_model
            # self.num_attn_heads = attn_num_heads
        else:
            if pre_proc.class_order or len(pre_proc.tgt_cnt) == 2:
                self.out_dim = 1
            else:
                self.out_dim = len(pre_proc.tgt_cnt)

            self.attention        = attention
            self.depth_attn_model = attn_depth_model
            self.num_attn_heads   = attn_num_heads
            self.bi_dir          = bi_dir
            self.dropout_prob    = dropout_prob
            self.embed_dim       = embed_dim
            self.n_rec_layers    = n_rec_layers
            self.lstm_layer_size = lstm_layer_size

            # Model Saver
            self.save_dir = save_dir or './checkpoints'
            if not os.path.exists(self.save_dir):
                os.makedirs(f"{self.save_dir}/best_f1")

            self.f1_report = f1_report or "macro"
        
            if save_params:
                params = (self.dropout_prob, self.embed_dim, self.n_rec_layers, self.lstm_layer_size, self.f1_report,
                        self.out_dim, self.save_dir, self.depth_attn_model, self.num_attn_heads)
                pickle.dump(params, open(f"{self.save_dir}/lstm_net.pkl", "wb"))
                
       
        self.best_f1 = 0
        self.build_graph()
        
    def build_graph(self):
        net_graph = tf.Graph()
        with net_graph.as_default():
            vocab_size = len(self.pre_proc.int_to_vocab)
            input_text, targets, lr, training = self.get_inputs()
            input_data_shape = tf.shape(input_text)
            cells, initial_states = self.get_init_cell(input_data_shape[0], self.lstm_layer_size, self.dropout_prob, self.lstm_layer_size, self.n_rec_layers, training, self.bi_dir)
            logits, final_states = self.build_nn(cells, self.lstm_layer_size, input_text, vocab_size, self.embed_dim, self.out_dim, training, self.bi_dir)
            
            
            # Loss function
            if self.out_dim > 2:
                cost = tf.compat.v1.losses.softmax_cross_entropy(targets, logits)
            else:
                cost = tf.compat.v1.losses.mean_squared_error(targets, logits)

            # Optimizer
            optimizer = tf.compat.v1.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)

        self.net_graph = net_graph
        self.input_text = input_text
        self.targets = targets
        self.lr = lr
        self.training = training
        self.fw_initial_state = initial_states[0]
        self.bw_initial_state = initial_states[1]
        self.logits = logits
        self.fw_final_state = final_states[0]
        self.bw_final_state = final_states[1]
        self.cost = cost
        self.train_op = train_op

    def destroy_graph(self):
        tf.compat.v1.reset_default_graph()

    def fit(self, train_x, train_y, batch_size = 128, early_stopping = False, learning_rate = 0.01, log_file=None, num_epochs = 100,
            random_state=None, show_every_epoch = True, test_x=None, test_y=None, valid_size=None):
        """
        Fit (Train) the model.
        :param x: train data
        :param y: train labels
        :param batch_size: Size of batches
        :param early_stopping: stops the training if the validation set performance gets no improvement
        :param learning_rate: Network Learning rate
        :param num_epochs: Number of Epochs for training
        :param random_state: seed to shuffle data when generating validation set
        :param show_every_n_batches: Show stats for every n number of batches
        :param test_x: test data
        :param text_y: test labels
        :param valid_size: The size of the validation set
        :return: Tuple (cell, initialize state)
        """
        valid_loss_list = []
        train_loss_list = []
        test_loss_list = []
        log_file = log_file or open(f"{self.save_dir}/log.txt", mode="a")
        worst_loss = 0
        best_loss_epoch = 0

        if early_stopping:
            train_x, valid_x, train_y, valid_y = net_batches.get_train_validation(x=train_x,
                                                                                  y=train_y,
                                                                                  valid_size=valid_size,
                                                                                  random_state=random_state)
            self.valid_loss = float("inf")

        batches = net_batches.get_batches(train_x, train_y, batch_size, whole_batch=False)

        with tf.compat.v1.Session(graph=self.net_graph) as sess:
            init_ = tf.compat.v1.global_variables_initializer()
            if self.has_pretr_emb:
                sess.run([init_, self.set_emb_wei], feed_dict={self.emb_ph: self.pre_tr_weights})
            else:
                sess.run(init_)
            
            
            for epoch_i in tqdm_notebook(range(num_epochs), desc="Epoch", leave=False):
                if self.has_pretr_emb:
                    fw_state, bw_state = sess.run([self.fw_initial_state, self.bw_initial_state],
                                                   {self.input_text: batches[0][0], self.emb_ph: self.pre_tr_weights})
                else:
                    fw_state, bw_state = sess.run([self.fw_initial_state, self.bw_initial_state],
                                                  {self.input_text: batches[0][0]})
                all_obs_tr_loss = []

                for batch_i, (x_, y_) in tqdm_notebook(enumerate(batches), desc="Train Batch", total=len(batches), leave=False):
                    feed = {self.input_text: x_,
                            self.targets: y_,
                            self.training: True,
                            self.fw_initial_state: fw_state,
                            self.bw_initial_state: bw_state,
                            self.lr: learning_rate}
                    if self.has_pretr_emb:
                        feed.update({self.emb_ph: self.pre_tr_weights})
                    train_loss, fw_state, bw_state, _ = sess.run([self.cost, self.fw_final_state, self.bw_final_state, self.train_op], feed)

                    all_obs_tr_loss += [train_loss*len(x_)]
                train_loss_list += [np.sum(all_obs_tr_loss)/len(train_x)]

                # Show every train and test performance every epoch
                if show_every_epoch:
                    print(f'Epoch {epoch_i:>3} Batch {batch_i:>4}/{len(batches)}   train_loss = {train_loss:.3f}', file=log_file)
                    
                    #Print test Loss and f1_score
                    if test_x and test_y:
                        test_batches = net_batches.get_batches(test_x, test_y, batch_size, whole_batch=False)
                        test_f1 = []
                        all_obs_te_loss = []
                        
                        if self.has_pretr_emb:
                            test_fw_state, test_bw_state = sess.run([self.fw_initial_state, self.bw_initial_state],
                                                                    {self.input_text: test_x, self.emb_ph: self.pre_tr_weights})
                        else:
                            test_fw_state, test_bw_state = sess.run([self.fw_initial_state, self.bw_initial_state],
                                                                    {self.input_text: test_x})
                                                                
                        obs_counter = 0
                        for x_t, y_t in tqdm_notebook(test_batches, desc="Test Batch", leave=False):
                            test_feed = {self.input_text: x_t,
                                        self.targets: y_t,
                                        self.training: False,
                                        self.fw_initial_state: test_fw_state,
                                        self.bw_initial_state: test_bw_state,
                                        self.lr: 0}
                            if self.has_pretr_emb:
                                feed.update({self.emb_ph: self.pre_tr_weights})
                            placeholders_list = [self.logits, self.cost, self.fw_final_state, self.bw_final_state]
                            test_logits, test_loss, test_fw_state, test_bw_state = sess.run(placeholders_list, test_feed)

                            test_f1.append(self.f1_score(test_logits, y_t)*len(y_t))
                            all_obs_te_loss += [test_loss*len(y_t)]
                            obs_counter += len(y_t)
                            
                        test_loss_list += [np.sum(all_obs_te_loss)/obs_counter]
                        test_f1 = np.sum(test_f1)/obs_counter
                        
                        print(f'\tTest_loss = {test_loss:.3f}    test_F1_Score = {test_f1:.3f}', file=log_file)

                if early_stopping:
                    if self.has_pretr_emb:
                        fw_init_state, bw_init_state = sess.run([self.fw_initial_state, self.bw_initial_state],
                                                                {self.input_text: valid_x, self.emb_ph: self.pre_tr_weights})
                    else:
                        fw_init_state, bw_init_state = sess.run([self.fw_initial_state, self.bw_initial_state],
                                                                {self.input_text: valid_x})
                    batch_feed = {self.input_text: valid_x,
                                  self.targets: valid_y,
                                  self.training: False,
                                  self.fw_initial_state: fw_init_state,
                                  self.bw_initial_state: bw_init_state,
                                  self.lr: 0}
                    if self.has_pretr_emb:
                        feed.update({self.emb_ph: self.pre_tr_weights})
                    placeholders_list = [self.logits, self.cost, self.fw_final_state, self.bw_final_state]
                    batch_logits, batch_loss, _, _ = sess.run(placeholders_list, batch_feed)
                    valid_f1 = self.f1_score(batch_logits, valid_y)
                    valid_loss_list += [batch_loss]

                    if batch_loss < self.valid_loss:
                        print(f"Validation loss decreased from {self.valid_loss:.4f} to {batch_loss:.4f}. Validation F1-score: {valid_f1:.4f}.", file=log_file)
                        self.valid_loss = batch_loss
                        best_loss_epoch = epoch_i
                        worst_loss = 0
                    else:
                        print(f"Validation loss increased from {self.valid_loss:.4f} to {batch_loss:.4f}. Validation F1-score: {valid_f1:.4f}.", file=log_file)
                        worst_loss += 1
                        if worst_loss >= early_stopping:
                            print("The training was stopped!", file=log_file)
                            break

                    if valid_f1 > self.best_f1:
                        self.best_f1 = valid_f1
                        self.save_model(sess, True)

                # Save Model at the end of each epoch
                self.save_model(sess)
                
        print(f"Best Test F1-Score: {self.best_f1:.3f}", file=log_file)
        log_file.flush()
        log_file.close()

        plt.plot(train_loss_list, label="Train Loss")
        plt.plot(test_loss_list, label="Test Loss")
        plt.plot(valid_loss_list, label="Valid Loss")
        plt.axvline(x=best_loss_epoch, linestyle="--", color="black")
        plt.legend()
        plt.show()

    def build_MultiHeadAttention(self, inputs, training):
        self.mha = MultiHeadAttention(self.depth_attn_model, self.num_attn_heads)
        mha_output, self.attention_weights = self.mha(inputs, inputs, inputs, None)
        mha_output = tf.keras.layers.Dropout(self.dropout_prob)(inputs=mha_output, training=training)
        return mha_output

    def build_nn(self, cells, rnn_size, input_data, vocab_size, embed_dim, out_dim, training, bi_dir):
        """
        Build part of the neural network
        :param cell: RNN cell
        :param rnn_size: Size of rnns
        :param input_data: Input data
        :param vocab_size: Vocabulary size
        :param embed_dim: Number of embedding dimensions
        :return: Tuple (Logits, FinalState)
        """
        # TODO: Implement Function
        embed_data = self.get_embed(input_data, vocab_size, embed_dim)
        
        lstm_output, final_state = self.build_rnn(cells, embed_data, training, bi_dir)
        self.lstm_output = lstm_output
        act_func = tf.nn.sigmoid if out_dim==1 else tf.nn.softmax
        if self.attention:
            attn_output = self.build_MultiHeadAttention(lstm_output, training)
            self.attn_output = attn_output
            logits = tf.keras.layers.Dense(out_dim)(attn_output)[:, -1, :]
        else:
            # Reshape output so it's a bunch of rows, one row for each step for each sequence.
            # Concatenate lstm_output over axis 1 (the columns)
            seq_output = tf.concat(values=lstm_output, axis=1)
            logits = tf.keras.layers.Dense(units=out_dim, activation=act_func)(inputs=seq_output)
        return logits, final_state

    def build_rnn(self, cells, inputs, training, bi_dir):
        """
        Create a RNN using a RNN Cell
        :param cell: RNN Cell
        :param inputs: Input text data
        :return: Tuple (Outputs, Final State)
        """
        # TODO: Implement Function
        if bi_dir:
            fw_layer = tf.keras.layers.RNN(cell=cells[0], return_sequences=self.attention, dtype=tf.float32, return_state=True)
            bw_layer = tf.keras.layers.RNN(cell=cells[1], return_sequences=self.attention, dtype=tf.float32, return_state=True, go_backwards=True)
            outputs, fw_state, bw_state = tf.keras.layers.Bidirectional(layer=fw_layer, backward_layer=bw_layer)(inputs=inputs, training=training)
            # fw_outputs = fw_outputs_state[0]
            # fw_state = fw_out_state[1:]
            fw_state = tf.expand_dims(tf.identity(fw_state), axis=0, name="fw_final_state")
            
            # bw_outputs = bw_outputs_state[0]
            # bw_state = bw_out_state[1:]
            bw_state = tf.expand_dims(tf.identity(bw_state), axis=0, name="bw_final_state")
            
            return outputs, [fw_state, bw_state]
        else:
            outputs_state = tf.keras.layers.RNN(cell=cells, return_sequences=self.attention, dtype=tf.float32, return_state=True)(inputs=inputs, training=training)
            outputs = outputs_state[0]
            state = outputs_state[1:]
            state = tf.identity(state, name="final_state")
            return outputs, state

    def f1_score(self, logits, targets):
        if len(logits) != len(targets):
            print(f"Logits length: {len(logits)}\t Targets length: {len(targets)}")

        if self.out_dim > 1:
            prds = [np.argmax(l) for l in logits]
            lbls = [np.argmax(t) for t in targets]
            return f1_score(lbls, prds, average=self.f1_report)
        else:
            prds = [np.round(l) for l in logits]
            # lbls = targets
            return f1_score(targets, prds, average=self.f1_report)

    def get_embed(self, input_data, vocab_size, embed_dim):
        """
        Create embedding for <input_data>.
        :param input_data: TF placeholder for text input.
        :param vocab_size: Number of words in vocabulary.
        :param embed_dim: Number of embedding dimensions
        :return: Embedded input.
        """
        # TODO: Implement Function
        self.pre_tr_weights = self.pre_proc.unpickle_weights()
        self.has_pretr_emb = isinstance(self.pre_tr_weights, np.ndarray)

        if self.has_pretr_emb:
            # embedding = tf.Variable(tf.constant(pre_tr_weights), trainable=False, name="emb_matrix")
            self.emb_ph = tf.compat.v1.placeholder(tf.float32, shape=list(self.pre_tr_weights.shape))
            embedding = tf.Variable(self.emb_ph, shape=list(self.pre_tr_weights.shape), trainable=False, name="emb_matrix")
            self.set_emb_wei = tf.assign(embedding, self.emb_ph, validate_shape=False)
        else:
            embedding = tf.Variable(tf.random.uniform((vocab_size, embed_dim),-1, 1)) # create embedding weight matrix here
        embed = tf.nn.embedding_lookup(embedding, input_data) # use tf.nn.embedding_lookup to get the hidden layer output
        
        return embed

    def get_init_cell(self, batch_size, rnn_size, drop_prob, lstm_size, n_layers, training, bi_dir):
        """
        Create an RNN Cell and initialize it.
        :param batch_size: Size of batches
        :param rnn_size: Size of RNNs
        :return: Tuple (cell, initialize state)
        """
        # TODO: Implement Function
        # Build the LSTM Cell
        def build_cell(num_units, drop_prob):
            # Use a basic LSTM cell
            lstm = tf.keras.layers.LSTMCell(num_units, dropout=drop_prob)

            # Add dropout to the cell outputs
            # drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            
            return lstm
        
        #create the forward NN 
        # Stack up multiple LSTM layers, for deep learning
        fw_cell = tf.keras.layers.StackedRNNCells([build_cell(lstm_size, drop_prob) for _ in range(n_layers)])
        fw_initial_state = fw_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        fw_initial_state = tf.identity(fw_initial_state, name="fw_initial_state")
        
        #create backwards NN
        if bi_dir:
            # Stack up multiple LSTM layers
            bw_cell = tf.keras.layers.StackedRNNCells([build_cell(lstm_size, drop_prob) for _ in range(n_layers)])
            bw_initial_state = bw_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
            bw_initial_state = tf.identity(bw_initial_state, name="bw_initial_state")

            return [fw_cell, bw_cell], [fw_initial_state, bw_initial_state]
        return fw_cell, fw_initial_state

    def get_inputs(self):
        """
        Create TF Placeholders for input, targets, and learning rate.
        :return: Tuple (input, targets, learning rate)
        """
        # TODO: Implement Function
        inputs = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="input")
        targets = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="targets")
        learn_rate = tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        training = tf.compat.v1.placeholder(tf.bool, shape=[], name="training")
        return inputs, targets, learn_rate, training

    def load_model(self, sess, best=False):
        var_list = None
        if self.has_pretr_emb:
            sess.run(tf.compat.v1.global_variables_initializer())
            var_list = []
            for var in self.net_graph.get_collection_ref("variables"):
                if var.op.name == "emb_matrix":
                    var_emb = var
                else:
                    var_list.append(var)

        saver = tf.compat.v1.train.Saver(save_relative_paths=True, var_list=var_list)
        if best:
            saver.restore(sess, tf.train.latest_checkpoint(f"{self.save_dir}/best_f1/"))
        else:
            saver.restore(sess, tf.train.latest_checkpoint(f"{self.save_dir}/"))

        if self.has_pretr_emb:
            var_emb = var_emb.assign(self.pre_proc.unpickle_weights())
    
    def predict(self, x, batch_size=128, attn_weights=False):
        predicted = ""

        with tf.compat.v1.Session(graph=self.net_graph) as sess:
            self.load_model(sess, best=True)
            batches = net_batches.get_batches(x, x, batch_size, whole_batch=False)
            if self.has_pretr_emb:
                fw_init_state, bw_init_state = sess.run([self.fw_initial_state, self.bw_initial_state],
                                                        {self.input_text: batches[0][0], self.emb_ph: self.pre_tr_weights})
            else:
                fw_init_state, bw_init_state = sess.run([self.fw_initial_state, self.bw_initial_state],
                                                        {self.input_text: batches[0][0]})

            for _, (x_batch, _) in enumerate(batches, 1):
                feed = {self.input_text: x_batch,
                        self.training: False,
                        self.fw_initial_state: fw_init_state,
                        self.bw_initial_state: bw_init_state,
                        self.lr: 0}
                if self.has_pretr_emb:
                    feed.update({self.emb_ph: self.pre_tr_weights})
                placeholders_list = [self.logits, self.attention_weights, self.attn_output]
                [batch_logits, attention_weights, attn_output] = sess.run(placeholders_list, feed)

                predicted = batch_logits if isinstance(predicted, str) else np.append(predicted, batch_logits, axis=0)
                predicted = np.clip(predicted, 0, 1)
                
        if attn_weights:
            return predicted, attention_weights, attn_output
        else:
            return predicted

    def save_model(self, sess, best=False):
        var_list = None
        if self.has_pretr_emb:
            var_list = [var for var in self.net_graph.get_collection_ref("variables") if var.op.name != "emb_matrix"]

        # Save Model
        saver = tf.compat.v1.train.Saver(save_relative_paths=True, var_list=var_list)
        if best:
            saver.save(sess, f"{self.save_dir}/best_f1/model", write_meta_graph=False)
        else:
            saver.save(sess, f"{self.save_dir}/model", write_meta_graph=False)

    def test(self, test_x, test_y, batch_size=128):
        test_f1 = []
        with tf.compat.v1.Session(graph=self.net_graph) as sess:
            self.load_model(sess, best=True)
            
            batches = net_batches.get_batches(test_x, test_y, batch_size, whole_batch=False)
            if self.has_pretr_emb:
                test_fw_state, test_bw_state = sess.run([self.fw_initial_state, self.bw_initial_state],
                                                        {self.input_text: batches[0][0], self.emb_ph: self.pre_tr_weights})
            else:
                test_fw_state, test_bw_state = sess.run([self.fw_initial_state, self.bw_initial_state],
                                                        {self.input_text: batches[0][0]})
            obs_counter = 0
            for _, (x, y) in enumerate(batches, 1):
                feed = {self.input_text: x,
                        self.targets: y,
                        self.fw_initial_state: test_fw_state,
                        self.bw_initial_state: test_bw_state,
                        self.lr: 0,
                        self.training: False}
                if self.has_pretr_emb:
                    feed.update({self.emb_ph: self.pre_tr_weights})
                batch_logits, test_fw_state, test_bw_state = sess.run([self.logits, self.fw_initial_state, self.bw_initial_state], feed_dict=feed)
                test_f1.append(self.f1_score(batch_logits, y)*len(y))
                obs_counter += len(y)
            print("Test F1-Score: {:.3f}".format(np.sum(test_f1)/obs_counter))