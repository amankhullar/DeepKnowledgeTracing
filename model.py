import numpy as np
import tensorflow as tf

class Config:
    """
    Initializing the hyperparameters that shall be used in the model
    """
    def __init___(self):
        self.n_questions
        self.n_hidden = 200
        self.dropout = 0.5
        self.batch_size = 32
        self.n_input = 2 * self.n_questions
        self.embed_size = 300
        self.lr = 0.001
        self.epsilon = 0.0001

class Model:
    """
    The Model to implement the Deep Knwoledge tracing using the Vanilla RNNs
    """
    config= Config()
    def add_placeholder(self):
        self.input_placeholder_X = tf.placeholder(dtype = tf.int32, shape = (None, 2 * self.n_questions), name = "input_x")
        self.input_placeholder_Y = tf.placeholder(dtype = tf.int32, shape = (None, self.n_questions), name = "input_y")
        self.labels_placeholder = tf.placeholder(dtype = tf.int32, shape = (None, self.n_questions), name = "labels")
        self.dropout_placeholder = tf.placeholder(dtype = tf.float32)

    def create_feed_dict(self, inputs_batch_x, inputs_batch_y, labels_batch = None, dropout = 1):
        feed_dict = {self.input_placeholder_x : inputs_batch_x, self.inputs_placeholder_y : inputs_batch_y, self.dropout_placeholder : dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self):
       embedded = tf.get_variable(name = "embedded", initializer = self.pretrained_embeddings)
       embeddings = tf.nn.embedding_lookup(params = embedded, ids = self.input_placeholder)
       return embeddings

