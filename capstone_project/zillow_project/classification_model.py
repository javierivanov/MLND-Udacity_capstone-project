import math
import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from zillow_project.data import Data, DataSet
from zillow_project.utils import xavier_init
from zillow_project.models import ZillowBaseModel

from sklearn.model_selection import train_test_split



class ClassificationModel(ZillowBaseModel):
    
    def run_default(self, data_file=None):
        self.set_dataset(n_classes=100, train_size=.7, test_size=.5, data_file=data_file)
        self.build_model()
        self.set_optimizer(learning_rate=.01)
        self.train_model(training_epochs=100, batch_size=64, logs_path='logs/classification/‘)


    def set_dataset(self, n_classes=100, train_size=.7, test_size=.5, data_file=None):
        data = pickle.load(open(data_file, 'rb'))
        data.set_targets()
        #n_classes = int(len(np.unique(data.y))*.25)
        self.n_classes = n_classes

        self.ymin = np.min(data.y)
        self.ymax = np.max(data.y)
        ydelta = self.ymax - self.ymin


        y = np.zeros((data.y.shape[0], self.n_classes))
        for i in range(len(y)):
            new_y = int((self.n_classes/ydelta)*(data.y[i] - self.ymin))
            if new_y >= self.n_classes: new_y = self.n_classes-1
            if new_y < 0: new_y = 0
            y[i,new_y] = 1

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
                layer_2 = tf.nn.relu(layer_2, name="relu_l2")
            # Hidden layer with RELU activation
            with tf.name_scope("Dense_layer"):
                layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['h3'])
                layer_3 = tf.nn.dropout(layer_3, .4)
                layer_3 = tf.nn.relu(layer_3, name="relu_l3")
            # Output layer with linear activation
            with tf.name_scope("Output_layer"):
                layer_out = tf.matmul(layer_3, weights['out']) + biases['out']
                self.model = tf.nn.sigmoid(layer_out, name="sigmoid_lout")

    def set_optimizer(self, learning_rate=.1):
        #MAE
        with tf.name_scope("mae"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.model, self.y)), name="mae")
        #ACCURACY
        with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.model, self.y), "float"), name="accuracy")
        #CAT ACCURACY
        correct_pred = tf.equal(tf.argmax(self.model,1), tf.argmax(self.y,1))
        self.categorical_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #MSE
        self.mse = tf.reduce_mean(tf.squared_difference(self.model, self.y))
        #CROSS ENTROPY
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.y))
        #CAT MAE
        ####
        #    Conversion of range of data
        #      Y = ((Y1-Y0)/(X1-X0)) * (X-X0) + Y0
        #
        #
        ymax = tf.constant([[self.ymax]]*self.n_input, tf.float64)
        ymin = tf.constant([[self.ymin]]*self.n_input, tf.float64)
        n_classes = tf.constant([[self.n_classes]]*self.n_input, tf.float64)

        pred_val = tf.cast(tf.argmax(self.model,1), tf.float64)
        y_val = tf.cast(tf.argmax(self.y,1), tf.float64)

        pred_realval = tf.add(tf.multiply(tf.divide(tf.subtract(self.ymax, self.ymin), self.n_classes), pred_val), self.ymin)
        y_realval = tf.add(tf.multiply(tf.divide(tf.subtract(self.ymax, self.ymin), self.n_classes), y_val), self.ymin)
        with tf.name_scope("categorical_mae"):
            self.categorical_mae = tf.reduce_mean(tf.abs(tf.subtract(pred_realval, y_realval)), name="categorical_mae")
        # Define loss and optimizer
        with tf.name_scope("optmizer"):
            self.global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = learning_rate
            self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 1000, .96, True, name="learning_rate")
            self.loss = self.mse
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step, name="minimize")


    def train_model(self, training_epochs=1000, batch_size=32, logs_path=None):
        # Initializing the variables
        init = tf.global_variables_initializer()
        tf.summary.scalar("mae_summary", self.mae)
        tf.summary.scalar("categorical_mae_summary", self.categorical_mae)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("learning_rate_summary", self.learning_rate)
        merged_summary = tf.summary.merge_all()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
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
                    c, _, lr = sess.run([self.loss, self.optimizer, self.learning_rate], feed_dict={self.x: batch_x,
                                                                self.y: batch_y})

                    
                    # Compute average loss
                    avg_cost += c / total_batch
                    # Upgrading progress bar data
                    pbar.set_description("Cost: {:.5f}, LR {:.5f}".format(avg_cost, lr))

                # Display logs per epoch step
                if True:
                    summary, _  = sess.run([merged_summary, self.categorical_mae], feed_dict=eval_valid_dict)
                    summary_writer.add_summary(summary, epoch)
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.6f}".format(avg_cost))
                    val_mae = self.categorical_mae.eval(eval_valid_dict)
                    train_mae = self.categorical_mae.eval(eval_train_dict)
                    test_mae = self.categorical_mae.eval(eval_test_dict)
                    print("VALIDATION ACC: {:.6f}".format(self.categorical_accuracy.eval(eval_valid_dict)))
                    print("VALIDATION MAE: {:.6f}".format(val_mae))
                    print("TESTING: ", test_mae)
                    print("TRAINING MAE: {:.6f}".format(train_mae))
                print()
            print("Optimization Finished!")
            print("TESTING: ", self.categorical_mae.eval(eval_test_dict))
