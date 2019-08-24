# author：hucheng

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import copy
import logging
import logging.config


class data_generation():
    '''
    Initialize data generator with paths to input files and setup variables
    '''
    def __init__(self, type, neg_number):
        print('init')
        self.data_type = type

        self.train_dataset = './data/' + self.data_type + '/' + self.data_type + '_train_dataset.csv'
        print('use train dataset ',self.train_dataset)
        self.test_dataset = './data/' + self.data_type + '/' + self.data_type + '_test_dataset.csv'
        print('use test dataset ',self.test_dataset)

        self.train_users = [] # each session in train_sessions has a corresponding 1 element in train_users
        # indicating the users
        self.train_sessions = []  # sessions with training items (removed prediction item), list with session lists
        self.train_items = []  # prediction items, one for each session
        self.train_neg_items = []  # Randomly sampled negative items, for each user (and session)
        self.train_pre_sessions = []  # Previous session collection, long-term item set L

        self.test_users = []
        self.test_candidate_items = []
        self.test_sessions = []
        self.test_pre_sessions = []
        self.test_real_items = []

        self.neg_number = neg_number
        self.user_number = 0
        self.item_number = 0
        self.train_batch_id = 0
        self.test_batch_id = 0
        self.records_number = 0

    def gen_train_data(self):
        '''
        Read in training data file
        split into sessions, and items per user.
        For each user-specific session, create negative items, create random item to predict
        :return:
        '''

        # read user and session columns whole dataset
        self.data = pd.read_csv(self.train_dataset, names=['user', 'sessions'], dtype='str')
        print('read-in size of training data ',self.data.shape)

        is_first_line = 1
        for line in self.data.values:
            if is_first_line:
                # the first line in file needs to contain the number of users in user row, and number of items in
                self.user_number = int(line[0]) # total number of users
                self.item_number = int(line[1]) # total number of items
                self.user_purchased_item = dict()  # Save each user's purchase record, which can be used for negative
                # sampling when train and rejected items for test
                is_first_line = 0
            else:
                user_id = int(line[0])
                # get list of sessions with items
                sessions = [i for i in line[1].split('@')]
                size = len(sessions)
                # do not consider users with less than 2 sessions
                if size < 2:
                    continue
                the_first_session = [int(i) for i in sessions[0].split(':')]
                self.train_pre_sessions.append(the_first_session)
                tmp = copy.deepcopy(the_first_session)
                self.user_purchased_item[user_id] = tmp
                # loop over all sessions after first session of user
                for j in range(1, size):
                    # save user for session j, we loop over sessions of user, so why is this inside j loop?
                    self.train_users.append(user_id)
                    # test = sessions[j].split(':')
                    current_session = [int(it) for it in sessions[j].split(':')]
                    # create list of negative items which are those not interacted with by user
                    neg = self.gen_neg(user_id)
                    self.train_neg_items.append(neg)
                    # Add the current session to the record purchased by the user The reason for this is because when
                    # you select the test item, you need to remove an item from the session. If you put it in the
                    # back, the current session is actually one less item used to make the current session prediction.
                    if j != 1:
                        # save list of previously purchased items for user
                        tmp = copy.deepcopy(self.user_purchased_item[user_id])
                        # this is the long-term item set L
                        self.train_pre_sessions.append(tmp)
                    tmp = copy.deepcopy(current_session)
                    # add items to list of items purchased by user, dictionary to lookup all items of user based on
                    # user name key
                    self.user_purchased_item[user_id].extend(tmp)

                    # Pick one randomly as a prediction item, remove that item from the training items
                    item = random.choice(current_session)
                    self.train_items.append(item)
                    current_session.remove(item)

                    # collect sessions for training
                    self.train_sessions.append(current_session)

                    # count the number of sessions
                    self.records_number += 1

    def gen_test_data(self):
        '''

        :return:
        '''
        # read test data
        self.data = pd.read_csv(self.test_dataset, names=['user', 'sessions'], dtype='str')
        print('read-in size of testing data ',self.data.shape)

        # test candidates are all (unique) items, based on total number of items item_number
        self.test_candidate_items = list(range(self.item_number))

        # Shuffle test data, by user
        sub_index = self.shuffle(len(self.data.values))
        data = self.data.values[sub_index]

        #
        for line in data:

            # first col is user id
            user_id = int(line[0])

            # consider only user if user was present in training set
            if user_id in self.user_purchased_item.keys():
                current_session = [int(i) for i in line[1].split(':')]
                # skip all users with less than 2 sessions
                if len(current_session) < 2:
                    continue
                self.test_users.append(user_id)

                # Pick one randomly as a prediction item, remove that item from the input items
                item = random.choice(current_session)
                self.test_real_items.append(int(item))
                current_session.remove(item)

                # input sessions/items
                self.test_sessions.append(current_session)
                self.test_pre_sessions.append(self.user_purchased_item[user_id])

        print('identified number of users with sessions: ',len(self.test_users))
        # batch_user = self.test_users[user_id:user_id + batch_size]
        # batch_item = self.test_candidate_items
        # batch_session = self.test_sessions[user_id]
        # batch_pre_session = self.test_pre_sessions[user_id]

    def shuffle(self, test_length):
        index = np.array(range(test_length))
        np.random.shuffle(index)
        sub_index = np.random.choice(index, int(test_length * 0.2))
        return sub_index

    def gen_neg(self, user_id):
        '''
        Create list of (negative) items which are those not purchased by user
        :param user_id: id of user
        :return:
        '''
        count = 0
        neg_item_set = list()
        while count < self.neg_number:
            neg_item = np.random.randint(self.item_number)
            if neg_item not in self.user_purchased_item[user_id]:
                neg_item_set.append(neg_item)
                count += 1
        return neg_item_set

    def gen_train_batch_data(self, batch_size):
        '''
        Get training data for session self.train_batch_id, including items to predict, negative list and previous items
        Get batch_size items to predict
        :param batch_size:
        :return:
        '''
        # l = len(self.train_users)

        if self.train_batch_id == self.records_number:
            self.train_batch_id = 0

        # even though multiple users, this is just one user, see comment for train_users
        batch_user = self.train_users[self.train_batch_id:self.train_batch_id + batch_size]
        # get items to predict, these items were extracted from multiple sessions
        batch_item = self.train_items[self.train_batch_id:self.train_batch_id + batch_size]
        # short-term item set S, just one session
        batch_session = self.train_sessions[self.train_batch_id]

        # batch_neg_item = self.train_neg_items[self.train_batch_id:self.train_batch_id + batch_size]

        # get negative item for session batch_session
        batch_neg_item = self.train_neg_items[self.train_batch_id]
        # get long-term item set L
        batch_pre_session = self.train_pre_sessions[self.train_batch_id]

        # increment train_batch_id to ...
        self.train_batch_id = self.train_batch_id + batch_size

        return batch_user, batch_item, batch_session, batch_neg_item, batch_pre_session

    def gen_test_batch_data(self, batch_size):
        '''
        Get test data for session self.test_batch_id
        :param batch_size:
        :return:
        '''

        l = len(self.test_users)

        if self.test_batch_id == l:
            self.test_batch_id = 0

        batch_user = self.test_users[self.test_batch_id:self.test_batch_id + batch_size]
        # batch_item is list of all items
        batch_item = self.test_candidate_items

        batch_session = self.test_sessions[self.test_batch_id]
        batch_pre_session = self.test_pre_sessions[self.test_batch_id]

        # increment test_batch_id
        self.test_batch_id = self.test_batch_id + batch_size

        return batch_user, batch_item, batch_session, batch_pre_session


class shan():
    # data_type :  TallM / GWL
    def __init__(self, data_type, neg_number, itera, global_dimension):
        '''

        :param data_type:
        :param neg_number: number of negative items used in prediction for each session
        :param itera:
        :param global_dimension: size of item and user embeddings
        '''

        print('init ... ')
        self.input_data_type = data_type

        # setup logging
        # Reads the logging configuration from file logging.conf
        logging.config.fileConfig('logging.conf')
        self.logger = logging.getLogger()
        # sends logging output to file
        print('send logging info to ','shan_log_' + data_type + '_d_' + str(global_dimension))
        fh = logging.FileHandler('shan_log_' + data_type + '_d_' + str(global_dimension), mode='a', encoding=None,
                                 delay=False)
        self.logger.addHandler(fh)

        # initialize data generator, with
        self.dg = data_generation(self.input_data_type, neg_number)

        # Data formatting
        print("reading training and testing data ....")
        self.dg.gen_train_data()
        self.dg.gen_test_data()
        print('all training and testing data was read')

        # dictionary to lookup all items purchased by user
        self.train_user_purchased_item_dict = self.dg.user_purchased_item

        # total number of users
        self.user_number = self.dg.user_number
        # total number of items
        self.item_number = self.dg.item_number
        # number of negative items
        self.neg_number = self.dg.neg_number

        # test data parameters
        self.test_users = self.dg.test_users
        self.test_candidate_items = self.dg.test_candidate_items
        self.test_sessions = self.dg.test_sessions
        self.test_pre_sessions = self.dg.test_pre_sessions
        self.test_real_items = self.dg.test_real_items

        # size of item and user embeddings
        self.global_dimension = global_dimension
        # batch size is 1! then get_test_data makes sense.., cannot be hyperparameter
        self.batch_size = 1
        self.results = []  # used to save the prediction results of each user of test, and finally calculate
        # precision

        self.step = 0
        self.iteration = itera
        # regularization parameter for user and item embedding parameters (see paper)
        self.lamada_u_v = 0.01
        # regularization parameter for weights of attention network
        self.lamada_a = 0.01

        # set normal distribution and parameters, as described in Algorithm 1 in paper
        self.initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
        # set uniform distribution and parameters, as described  in Algorithm 1 in paper
        self.initializer_param = tf.random_uniform_initializer(minval=-np.sqrt(3 / self.global_dimension),
                                                               maxval=np.sqrt(3 / self.global_dimension))

        # user id and item id for training input
        self.user_id = tf.placeholder(tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(tf.int32, shape=[None], name='item_id')

        # further training input
        # Whether it is the current session or the previous session collection, it is an array in the data processing
        # stage, and the array content is the item number.
        self.current_session = tf.placeholder(tf.int32, shape=[None], name='current_session')
        self.pre_sessions = tf.placeholder(tf.int32, shape=[None], name='pre_sessions')
        self.neg_item_id = tf.placeholder(tf.int32, shape=[None], name='neg_item_id')

        # user embedding U
        self.user_embedding_matrix = tf.get_variable('user_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.user_number, self.global_dimension])
        # item embedding V
        self.item_embedding_matrix = tf.get_variable('item_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.item_number, self.global_dimension])
        # W_1 in MLP 1 of attention network 1 (paper)
        self.the_first_w = tf.get_variable('the_first_w', initializer=self.initializer_param,
                                           shape=[self.global_dimension, self.global_dimension])
        # W_2 in MLP 2
        self.the_second_w = tf.get_variable('the_second_w', initializer=self.initializer_param,
                                            shape=[self.global_dimension, self.global_dimension])
        # b_1 bias of MLP 1
        self.the_first_bias = tf.get_variable('the_first_bias', initializer=self.initializer_param,
                                              shape=[self.global_dimension])
        # b_2 bias of MLP 2
        self.the_second_bias = tf.get_variable('the_second_bias', initializer=self.initializer_param,
                                               shape=[self.global_dimension])

    def attention_level_one(self, user_embedding, pre_sessions_embedding, the_first_w, the_first_bias):
        '''
        First attention level which computes the long-term user representation u^long
        :param user_embedding: u
        :param pre_sessions_embedding: V_j
        :param the_first_w: W_1 in MLP
        :param the_first_bias: b_1 in MLP
        :return:
        '''

        # Matmul and multiply methods require dimensional changes due to dimensions
        # a matrix with a final weight of 1*n
        # h_1_j = RELU(pre_sessions_embedding*the_first_w + the_first_bias) (eq 1 in paper)
        # a_j = SOFTMAX((h_1_j * user_embedding^T)^T) (eq 2)
        self.weight = tf.nn.softmax(tf.transpose(tf.matmul(tf.nn.relu(
            tf.add(tf.matmul(pre_sessions_embedding, the_first_w), the_first_bias)), tf.transpose(user_embedding))))

        # u^long = SUM(pre_sessions_embedding * weight^T) (eq 3 in paper)
        out = tf.reduce_sum(tf.multiply(pre_sessions_embedding, tf.transpose(self.weight)), axis=0)

        return out

    def attention_level_two(self, user_embedding, long_user_embedding, current_session_embedding, the_second_w,
                            the_second_bias):
        '''

        :param user_embedding:
        :param long_user_embedding: long-term user embedding from attention_level_one
        :param current_session_embedding: x_j
        :param the_second_w:
        :param the_second_bias:
        :return:
        '''
        # Need to add long_user_embedding to current_session_embedding for attention, The paper states that the
        # representation of long_user_embedding will not change according to the parameters calculated by softmax.

        # first construct x_j based on current_session_embedding and long_user_embedding, why is concat order the
        # opposite as in paper?
        # then perform same steps as in attention_level_one (eq. 4,5)
        self.weight = tf.nn.softmax(tf.transpose(tf.matmul(
            tf.nn.relu(tf.add(
                tf.matmul(tf.concat([current_session_embedding, tf.expand_dims(long_user_embedding, axis=0)], 0),
                          the_second_w),
                the_second_bias)), tf.transpose(user_embedding))))

        # hybrid user representation u^hybrid based on (eq. 6)
        out = tf.reduce_sum(
            tf.multiply(tf.concat([current_session_embedding, tf.expand_dims(long_user_embedding, axis=0)], 0),
                        tf.transpose(self.weight)), axis=0)
        return out

    def build_model(self):
        '''
        Combine network elements to build final model. Setup network propagation and final loss function for specific
        session input.
         Saves the items with the highest preference score :return:
        '''
        print('building model ... ')

        # look up embedding for user_id
        self.user_embedding = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_id)
        # lookup embedding for item_id
        self.item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.item_id)
        # lookup other items
        self.current_session_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.current_session)
        self.pre_sessions_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.pre_sessions)
        self.neg_item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.neg_item_id)

        # long-term user embedding u^l
        self.long_user_embedding = self.attention_level_one(self.user_embedding, self.pre_sessions_embedding,
                                                            self.the_first_w, self.the_first_bias)
        #  hybrid user representation u^hybrid
        self.hybrid_user_embedding = self.attention_level_two(self.user_embedding, self.long_user_embedding,
                                                              self.current_session_embedding,
                                                              self.the_second_w, self.the_second_bias)

        # compute preference R = u_t * v_j (eq. 7)
        self.positive_element_wise = tf.matmul(tf.expand_dims(self.hybrid_user_embedding, axis=0),
                                               tf.transpose(self.item_embedding))
        # compute preference R for negative items
        self.negative_element_wise = tf.matmul(tf.expand_dims(self.hybrid_user_embedding, axis=0),
                                               tf.transpose(self.neg_item_embedding))

        # calculate loss through MAP (eq. 9)
        self.intention_loss = tf.reduce_mean(
            -tf.log(tf.nn.sigmoid(self.positive_element_wise - self.negative_element_wise)))
        # regularization terms based on lambda * user and item embeddings and l2 norm
        self.regular_loss_u_v = tf.add(tf.add(self.lamada_u_v * tf.nn.l2_loss(self.user_embedding),
                                              self.lamada_u_v * tf.nn.l2_loss(self.item_embedding)),
                                       self.lamada_u_v * tf.nn.l2_loss(self.neg_item_embedding))
        # regularization term bsaed on the MLP weights
        self.regular_loss_a = tf.add(self.lamada_a * tf.nn.l2_loss(self.the_first_w),
                                     self.lamada_a * tf.nn.l2_loss(self.the_second_w))
        # sum regularization terms
        self.regular_loss = tf.add(self.regular_loss_a, self.regular_loss_u_v)

        # final loss function
        self.intention_loss = tf.add(self.intention_loss, self.regular_loss)

        # Increase the test operation. Since the length of each user's pre_sessions and current_session is different,
        # you cannot use the same matrix to represent the simultaneous calculation. Therefore, each user is
        # calculated once, the result is retained and statistical attention is paid. The entire item_embeeding of the
        # test set is obtained. [M*K] matrix, M is the number of all items, K is the dimension

        # Finds values of preference R and indices of the k largest entries for the last dimension.
        self.top_value_10, self.top_index_10 = tf.nn.top_k(self.positive_element_wise, k=10, sorted=True)
        self.top_value_20, self.top_index_20 = tf.nn.top_k(self.positive_element_wise, k=20, sorted=True)
        self.top_value_50, self.top_index_50 = tf.nn.top_k(self.positive_element_wise, k=50, sorted=True)

    def run(self):
        '''
        Do the computation
        :return:
        '''
        print('running ... ')
        with tf.Session() as self.sess:

            # use gradient descent to minimize loss (no global_step applied)
            self.intention_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(
                self.intention_loss)

            init = tf.global_variables_initializer()
            self.sess.run(init)

            for iter in range(self.iteration):
                print('start iteration/epoch ',iter)
                self.logger.info('iteration/epoch: '+str(iter))

                all_loss = 0

                # go through all sessions in training set
                while self.step * self.batch_size < self.dg.records_number:

                    # Read train data by batch/1 session
                    batch_user, batch_item, batch_session, batch_neg_item, batch_pre_sessions = \
                        self.dg.gen_train_batch_data(self.batch_size)

                    # one step with 1 session
                    _, loss = self.sess.run([self.intention_optimizer, self.intention_loss],
                                            feed_dict={self.user_id: batch_user,
                                                       self.item_id: batch_item,
                                                       self.current_session: batch_session,
                                                       self.neg_item_id: batch_neg_item,
                                                       self.pre_sessions: batch_pre_sessions
                                                       })
                    # accumulate final loss
                    all_loss += loss
                    self.step += 1
                    # if self.step * self.batch_size % 5000 == 0:
                self.logger.info('loss after iteration/epoch = '+str(all_loss)+'\n')
                self.logger.info('eval ...')

                # evaluate performance on test set, calculatione Recall@k and MRR@k
                self.test_set_evaluation()

                print(self.step, '/', self.dg.train_batch_id, '/', self.dg.records_number)
                self.step = 0

    def P_k(self, pre_top_k, true_items):
        '''
        Calculat recall@k
        :param pre_top_k:
        :param true_items: true items (from labels)
        :return:
        '''
        right_pre = 0
        user_number = len(pre_top_k)
        # loop over session predictions
        for i in range(user_number):
            if true_items[i] in pre_top_k[i][0]:
                right_pre += 1
        return right_pre / user_number

    def MRR_k(self, pre_top_k, true_items):
        '''
        Mean reciprocal rank MRR@k
        :param pre_top_k:
        :param true_items:
        :return:
        '''
        MRR_rate = 0
        user_number = len(pre_top_k)
        for i in range(user_number):
            if true_items[i] in pre_top_k[i][0]:
                # get rank
                index = pre_top_k[i].tolist()[0].index(true_items[i])
                MRR_rate += 1 / (index + 1)
        return MRR_rate / user_number

    def test_set_evaluation(self):
        '''

        :return:
        '''
        pre_top_k_10 = []
        pre_top_k_20 = []
        pre_top_k_50 = []

        # iterate over users in test set
        for _ in self.test_users:
            # get test data session
            batch_user, batch_item, batch_session, batch_pre_session = self.dg.gen_test_batch_data(
                self.batch_size)

            # calculate preference for lists of top 10, 20, 50 items
            top_index_10, top_index_20, top_index_50 = self.sess.run(
                [self.top_index_10, self.top_index_20, self.top_index_50],
                feed_dict={self.user_id: batch_user,
                           self.item_id: batch_item,
                           self.current_session: batch_session,
                           self.pre_sessions: batch_pre_session})

            # append results
            pre_top_k_10.append(top_index_10)
            pre_top_k_20.append(top_index_20)
            pre_top_k_50.append(top_index_50)

        # evaluate
        P_10 = self.P_k(pre_top_k_10, self.test_real_items)
        MRR_10 = self.MRR_k(pre_top_k_10, self.test_real_items)

        P_20 = self.P_k(pre_top_k_20, self.test_real_items)
        MRR_20 = self.MRR_k(pre_top_k_20, self.test_real_items)

        P_50 = self.P_k(pre_top_k_50, self.test_real_items)
        MRR_50 = self.MRR_k(pre_top_k_50, self.test_real_items)

        self.logger.info(self.input_data_type + ',' + 'Recall@10' + ' = ' + str(P_10))
        self.logger.info(self.input_data_type + ',' + 'MRR@10' + ' = ' + str(MRR_10) + '\n')

        self.logger.info(self.input_data_type + ',' + 'Recall@20' + ' = ' + str(P_20))
        self.logger.info(self.input_data_type + ',' + 'MRR@20' + ' = ' + str(MRR_20) + '\n')

        self.logger.info(self.input_data_type + ',' + 'Recall@50' + ' = ' + str(P_50))
        self.logger.info(self.input_data_type + ',' + 'MRR@50' + ' = ' + str(MRR_50) + '\n')

        return


if __name__ == '__main__':

    # specify type of data set, types of data sets available
    type = ['tallM', 'gowalla', 'lastFM', 'fineFoods', 'movieLens', 'tafeng']
    index = 0
    print('use dataset ',type[index])

    # hyperparameters
    # number of negative items for training, per session
    neg_number = 10
    # number of epochs
    itera = 100
    # size of item and user embeddings
    global_dimension = 50

    # init model
    model = shan(type[index], neg_number, itera, global_dimension)

    # build model
    model.build_model()

    # start training and evaluation
    model.run()
