import math
import pickle

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from zillow_project.data import Data
from zillow_project.utils import xavier_init
from zillow_project.models import ZillowBaseModel

from sklearn.model_selection import train_test_split

class MLPRegressionModel(ZillowBaseModel):
    
    def run_default(self, data_file=None):
        self.set_dataset(n_classes=1, train_size=.7, test_size=.5, data_file=None)
        self.build_model()
        self.set_optimizer(learning_rate=.01)
        self.train_model(training_epochs=100, batch_size=64, logs_path='logs/regression/')

    def set_dataset(self, n_classes=1, train_size=.8, test_size=.5, data_file=None):        
        data = pickle.load(open(data_file, 'rb'))
        data.set_targets()
        y = data.y.reshape(data.y.shape[0], 1)
        self.n_classes = n_classes

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(data.X, y, test_size=1.0-train_size, random_state=22)
        self.valid_x, self.test_x, self.valid_y, self.test_y = train_test_split(self.test_x, self.test_y, test_size=test_size, random_state=23)
        self.n_input = self.train_x.shape[1]
        print("Using, ", self.n_input, "features.")

    def build_model(self):
        # tf Graph input
        with tf.name_scope('model'):
            self.x = tf.placeholder("float", [None, self.n_input],'input_x')
            self.y = tf.placeholder("float", [None, self.n_classes], 'input_y')
            n_hidden_1 = 4096
            n_hidden_2 = 4096
            n_hidden_3 = 2048
            # Create model
            weights = {
                'h1': tf.Variable(xavier_init(self.n_input, n_hidden_1), name='wh1'),
                'h2': tf.Variable(xavier_init(n_hidden_1, n_hidden_2), name="wh2"),
                'h3': tf.Variable(xavier_init(n_hidden_2, n_hidden_3), name="wh3"),
                'out': tf.Variable(xavier_init(n_hidden_3, self.n_classes), name="wo")
            }
            biases = {
                'h1': tf.Variable(tf.zeros([n_hidden_1]), name="bh1"),
                'h2': tf.Variable(tf.zeros([n_hidden_2]), name="bh2"),
                'h3': tf.Variable(tf.zeros([n_hidden_3]), name="bh3"),
                'out': tf.Variable(tf.zeros([self.n_classes]), name="bo")
            }
            # Hidden layer with RELU activation
            with tf.name_scope("Dense_layer"):
                layer_1 = tf.add(tf.matmul(self.x, weights['h1']), biases['h1'])
                layer_1 = tf.nn.dropout(layer_1, .4)
                layer_1 = tf.nn.relu(layer_1, name='relu_l1')
            # Hidden layer with RELU activation
            with tf.name_scope("Dense_layer"):
                layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
                layer_2 = tf.nn.dropout(layer_2, .4)
                layer_2 = tf.nn.sigmoid(layer_2, name="sigmoid_l2")
            # Hidden layer with RELU activation
            with tf.name_scope("Dense_layer"):
                layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['h3'])
                layer_3 = tf.nn.dropout(layer_3, .4)
                layer_3 = tf.nn.tanh(layer_3, name="tanh_l3")
            # Output layer with linear activation
            with tf.name_scope("output_layer"):
                self.model = tf.matmul(layer_3, weights['out']) + biases['out']
    def set_optimizer(self, learning_rate=0.1):
        #Mean Absolute Error
        with tf.name_scope("mae"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.model, self.y)), name='mae')

        #Mean Squaered Error
        with tf.name_scope("mse"):
            self.mse = tf.reduce_mean(tf.squared_difference(self.model, self.y), name='mse')

        # Define loss and optimizer
        with tf.name_scope("optimizer"):
            self.global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = learning_rate
            self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 500, .96, True, name='learning_rate')
            self.loss = self.mse
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step, name='minimize')


    def train_model(self, training_epochs=1000, batch_size=32, logs_path=None):
        # Initializing the variables
        init = tf.global_variables_initializer()
        tf.summary.scalar("mae_summary", self.mae)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("learning_rate_summary", self.learning_rate)
        merged_summary = tf.summary.merge_all()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            #Getting data
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
            #eval train dict
            eval_train_dict = {self.x: self.train_x, self.y: self.train_y}
            #eval train dict
            eval_valid_dict = {self.x: self.valid_x, self.y: self.valid_y}
            #eval test dict
            eval_test_dict = {self.x: self.test_x, self.y: self.test_y}

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = math.ceil(self.train_x.shape[0]/batch_size)
                # Loop over all batches
                pbar = tqdm(range(total_batch))
                for i in pbar:
                    batch_x = self.train_x[i*batch_size:i*batch_size+batch_size]
                    batch_y = self.train_y[i*batch_size:i*batch_size+batch_size]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c, lr = sess.run([self.optimizer, self.loss, self.learning_rate], feed_dict={self.x: batch_x,
                                                                self.y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                    # Upgrading progress bar data
                    pbar.set_description("Cost: {:.5f}, LR: {}".format(avg_cost, lr))

                # Display logs per epoch step
                if True:
                    summary, _  = sess.run([merged_summary, self.mae], feed_dict=eval_valid_dict)
                    summary_writer.add_summary(summary, epoch)
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.6f}".format(avg_cost))
                    train_mae = self.mae.eval(eval_train_dict)
                    val_mae = self.mae.eval(eval_valid_dict)
                    test_mae = self.mae.eval(eval_test_dict)
                    print("VALIDATION MAE: {:.6f}".format(val_mae))
                    print("TRAINING MAE: {:.6f}".format(train_mae))
                    print("TESTING: ", test_mae)
                print()
            print("Optimization Finished!")
            print("TESTING: ", self.mae.eval(eval_test_dict))
