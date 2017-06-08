import tensorflow as tf


class AttentionSeq2Seq(object):

    def __init__(self,
                 sess,
                 batch_size,
                 encoder_sequence_size,
                 decoder_sequence_size,
                 encoder_symbol_size,
                 decoder_symbol_size,
                 embedding_size,
                 hidden_size):

        self.batch_size = batch_size
        self.encoder_sequence_size = encoder_sequence_size
        self.decoder_sequence_size = decoder_sequence_size
        self.encoder_symbol_size = encoder_symbol_size
        self.decoder_symbol_size = decoder_symbol_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.sess = sess
        self._build_train_net()

    def encoder(self):
        self.enc_batch_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.encoder_sequence_size))

        with tf.variable_scope('encoder') as scope:
            enc_embeddings = tf.Variable(tf.random_normal([self.encoder_symbol_size,
                                                            self.embedding_size],
                                                            stddev=1.0))
            batch_embeddings = tf.nn.embedding_lookup(enc_embeddings, self.enc_batch_inputs)
            fw_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
            bw_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)

            fw_initial_state = fw_cell.zero_state(self.batch_size, tf.float32)
            bw_initial_state = bw_cell.zero_state(self.batch_size, tf.float32)
            '''
            outputs, states = tf.nn.dynamic_rnn(cell, 
                                                batch_embeddings, 
                                                initial_state=initial_state, 
                                                dtype=tf.float32)
            '''
            outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                              bw_cell,
                                                              batch_embeddings,
                                                              initial_state_fw=fw_initial_state,
                                                              initial_state_bw=bw_initial_state,
                                                              dtype=tf.float32)

        self.context_vectors = tf.concat([outputs[0], outputs[1]], axis=2)
        self.encoder_hidden_state = states[1]

        return outputs, states

    @staticmethod
    def attention(encoder_hidden_states,
                  last_hidden_state):

        encoder_seq_len = encoder_hidden_states.get_shape().as_list()[1]
        encoder_hidden_state_dim = encoder_hidden_states.get_shape().as_list()[2]
        last_hidden_state_dim = last_hidden_state.get_shape().as_list()[1]

        with tf.variable_scope('attention') as scope:
            _encoder_hidden_states = tf.reshape(encoder_hidden_states, shape=[-1, encoder_hidden_state_dim])
            W_a = tf.Variable(tf.random_normal([last_hidden_state_dim,
                                                 last_hidden_state_dim],
                                                stddev=1.0), name='W_a')
            U_a = tf.Variable(tf.random_normal([encoder_hidden_state_dim,
                                                 last_hidden_state_dim],
                                                stddev=1.0), name='U_a')
            v_a = tf.Variable(tf.random_normal([last_hidden_state_dim, 1],
                                                stddev=1.0), name='v_a')
            b = tf.Variable(tf.zeros(shape=[last_hidden_state_dim]), name='b')

            product_hidden_states = tf.reshape(tf.matmul(_encoder_hidden_states, U_a),
                                                 shape=[-1, encoder_seq_len, last_hidden_state_dim])
            product_last_state = tf.expand_dims(tf.matmul(last_hidden_state, W_a), dim=1)
            signals = tf.reshape(tf.nn.tanh(product_hidden_states + product_last_state + b),
                                 shape=[-1, last_hidden_state_dim])
            alignment = tf.reshape(tf.matmul(signals, v_a), [-1, encoder_seq_len], name='alignment')
            alpha = tf.nn.softmax(alignment, name='alpha')
            context = tf.matmul(tf.transpose(tf.expand_dims(alpha, dim=2), perm=[0, 2, 1]), encoder_hidden_states,
                                name='context')
            context = tf.reshape(context, [-1, encoder_hidden_state_dim])

        return context

    def decoder_train(self):
        self.dec_batch_inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.decoder_sequence_size))
        self.batch_labels = tf.placeholder(tf.int32, shape=(self.batch_size, self.decoder_sequence_size))

        with tf.variable_scope('decoder') as scope:
            self.decoder_scope = scope
            embeddings = tf.get_variable('embeddings',
                                         [self.decoder_symbol_size, self.embedding_size],
                                         initializer=tf.random_uniform_initializer())
            dec_batch_embeddings = tf.nn.embedding_lookup(embeddings, self.dec_batch_inputs)

            self.cells = [tf.contrib.rnn.GRUCell(num_units=self.hidden_size) for _ in range(self.decoder_sequence_size)]
            initial_state = self.encoder_hidden_state
            outputs_list = []

            self.decoder_time_step_scope = []
            for t in range(self.decoder_sequence_size):
                with tf.variable_scope('decoder_' + str(t) + '_step') as t_scope:
                    self.decoder_time_step_scope.append(t_scope)
                    context = self.attention(self.context_vectors, initial_state)
                    concat_batch_inputs = tf.expand_dims(tf.concat([dec_batch_embeddings[:, t], context], 1),
                                                         dim=1)
                    cell = self.cells[t]
                    output, state = tf.nn.dynamic_rnn(cell,
                                                      concat_batch_inputs,
                                                      initial_state=initial_state,
                                                      dtype=tf.float32)
                    inputs_for_fc = tf.reshape(output, [-1, self.hidden_size])
                    fc_outputs = tf.contrib.layers.fully_connected(inputs=inputs_for_fc,
                                                                   num_outputs=self.decoder_symbol_size)
                    outputs_list.append(tf.reshape(fc_outputs, [-1, self.decoder_symbol_size]))
                    initial_state = state

            logits = tf.stack(outputs_list, axis=1)
            weights = tf.ones([self.batch_size, self.decoder_sequence_size])
            sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                             targets=self.batch_labels,
                                                             weights=weights)
            self.loss = tf.reduce_mean(sequence_loss)

    def decoder_pred(self):
        self.current_word = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

        with tf.variable_scope(self.decoder_scope, reuse=True):
            embeddings = tf.get_variable('embeddings')
            initial_state = self.encoder_hidden_state
            outputs_list = []

            for t in range(self.decoder_sequence_size):
                dec_batch_embeddings = tf.nn.embedding_lookup(embeddings, self.current_word)

                with tf.variable_scope(self.decoder_time_step_scope[t], reuse=True):
                    context = self.attention(self.context_vectors, initial_state)
                    concat_batch_inputs = tf.expand_dims(tf.concat([dec_batch_embeddings[:, 0], context], 1),
                                                         dim=1)
                    cell = self.cells[t]
                    output, state = tf.nn.dynamic_rnn(cell,
                                                      concat_batch_inputs,
                                                      initial_state=initial_state,
                                                      dtype=tf.float32)
                    inputs_for_fc = tf.reshape(output, [-1, self.hidden_size])
                    fc_outputs = tf.contrib.layers.fully_connected(inputs=inputs_for_fc,
                                                                   num_outputs=self.decoder_symbol_size)
                    logits = tf.reshape(fc_outputs, [self.batch_size, 1, self.decoder_symbol_size])
                    current_word = tf.argmax(logits, axis=2)
                    outputs_list.append(current_word)
                    initial_state = state

            self.pred = tf.stack(outputs_list, axis=1)

    def _build_train_net(self):
        self.encoder()
        self.decoder_train()
        self.decoder_pred()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss)


    def train(self, encoder_data, decoder_data, label_data):
        return self.sess.run([self.loss, self.optimizer], feed_dict={self.enc_batch_inputs: encoder_data,
                                                                     self.dec_batch_inputs: decoder_data,
                                                                     self.batch_labels: label_data})

    def prediction(self, encoder_data, starting_data):
        return self.sess.run(self.pred, feed_dict={self.enc_batch_inputs: encoder_data,
                                                   self.current_word: starting_data})
