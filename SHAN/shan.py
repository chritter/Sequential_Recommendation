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
        # batch size is 1! then get_test_data makes sense..
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

        :param user_embedding:
        :param pre_sessions_embedding:
        :param the_first_w:
        :param the_first_bias:
        :return:
        '''

        # 由于维度的原因，matmul和multiply方法要维度的变化
        # 最终weight为 1*n 的矩阵
        self.weight = tf.nn.softmax(tf.transpose(tf.matmul(tf.nn.relu(
            tf.add(tf.matmul(pre_sessions_embedding, the_first_w), the_first_bias)), tf.transpose(user_embedding))))

        out = tf.reduce_sum(tf.multiply(pre_sessions_embedding, tf.transpose(self.weight)), axis=0)
        return out

    def attention_level_two(self, user_embedding, long_user_embedding, current_session_embedding, the_second_w,
                            the_second_bias):
        # 需要将long_user_embedding加入到current_session_embedding中来进行attention，
        # 论文中规定，long_user_embedding的表示也不会根据softmax计算得到的参数而变化。

        self.weight = tf.nn.softmax(tf.transpose(tf.matmul(
            tf.nn.relu(tf.add(
                tf.matmul(tf.concat([current_session_embedding, tf.expand_dims(long_user_embedding, axis=0)], 0),
                          the_second_w),
                the_second_bias)), tf.transpose(user_embedding))))
        out = tf.reduce_sum(
            tf.multiply(tf.concat([current_session_embedding, tf.expand_dims(long_user_embedding, axis=0)], 0),
                        tf.transpose(self.weight)), axis=0)
        return out

    def build_model(self):
        print('building model ... ')
        self.user_embedding = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_id)
        self.item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.item_id)
        self.current_session_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.current_session)
        self.pre_sessions_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.pre_sessions)
        self.neg_item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.neg_item_id)

        self.long_user_embedding = self.attention_level_one(self.user_embedding, self.pre_sessions_embedding,
                                                            self.the_first_w, self.the_first_bias)

        self.hybrid_user_embedding = self.attention_level_two(self.user_embedding, self.long_user_embedding,
                                                              self.current_session_embedding,
                                                              self.the_second_w, self.the_second_bias)

        # compute preference
        self.positive_element_wise = tf.matmul(tf.expand_dims(self.hybrid_user_embedding, axis=0),
                                               tf.transpose(self.item_embedding))
        self.negative_element_wise = tf.matmul(tf.expand_dims(self.hybrid_user_embedding, axis=0),
                                               tf.transpose(self.neg_item_embedding))
        self.intention_loss = tf.reduce_mean(
            -tf.log(tf.nn.sigmoid(self.positive_element_wise - self.negative_element_wise)))
        self.regular_loss_u_v = tf.add(tf.add(self.lamada_u_v * tf.nn.l2_loss(self.user_embedding),
                                              self.lamada_u_v * tf.nn.l2_loss(self.item_embedding)),
                                       self.lamada_u_v * tf.nn.l2_loss(self.neg_item_embedding))
        self.regular_loss_a = tf.add(self.lamada_a * tf.nn.l2_loss(self.the_first_w),
                                     self.lamada_a * tf.nn.l2_loss(self.the_second_w))
        self.regular_loss = tf.add(self.regular_loss_a, self.regular_loss_u_v)
        self.intention_loss = tf.add(self.intention_loss, self.regular_loss)

        # 增加test操作，由于每个用户pre_sessions和current_session的长度不一样，
        # 所以无法使用同一个矩阵进行表示同时计算，因此每个user计算一次，将结果保留并进行统计
        # 注意，test集合的整个item_embeeding得到的是 [M*K]的矩阵，M为所有item的个数，K为维度
        self.top_value_10, self.top_index_10 = tf.nn.top_k(self.positive_element_wise, k=10, sorted=True)
        self.top_value_20, self.top_index_20 = tf.nn.top_k(self.positive_element_wise, k=20, sorted=True)
        self.top_value_50, self.top_index_50 = tf.nn.top_k(self.positive_element_wise, k=50, sorted=True)

    def run(self):
        print('running ... ')
        with tf.Session() as self.sess:
            self.intention_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(
                self.intention_loss)
            init = tf.global_variables_initializer()
            self.sess.run(init)

            for iter in range(self.iteration):
                print('new iteration begin ... ')
                self.logger.info('iteration: '+str(iter))

                all_loss = 0
                while self.step * self.batch_size < self.dg.records_number:
                    # 按批次读取数据
                    batch_user, batch_item, batch_session, batch_neg_item, batch_pre_sessions = self.dg.gen_train_batch_data(
                        self.batch_size)

                    _, loss = self.sess.run([self.intention_optimizer, self.intention_loss],
                                            feed_dict={self.user_id: batch_user,
                                                       self.item_id: batch_item,
                                                       self.current_session: batch_session,
                                                       self.neg_item_id: batch_neg_item,
                                                       self.pre_sessions: batch_pre_sessions
                                                       })
                    all_loss += loss
                    self.step += 1
                    # if self.step * self.batch_size % 5000 == 0:
                self.logger.info('loss = '+str(all_loss)+'\n')
                self.logger.info('eval ...')
                self.evolution()
                print(self.step, '/', self.dg.train_batch_id, '/', self.dg.records_number)
                self.step = 0

    def P_k(self, pre_top_k, true_items):
        right_pre = 0
        user_number = len(pre_top_k)
        for i in range(user_number):
            if true_items[i] in pre_top_k[i][0]:
                right_pre += 1
        return right_pre / user_number

    def MRR_k(self, pre_top_k, true_items):
        MRR_rate = 0
        user_number = len(pre_top_k)
        for i in range(user_number):
            if true_items[i] in pre_top_k[i][0]:
                index = pre_top_k[i].tolist()[0].index(true_items[i])
                MRR_rate += 1 / (index + 1)
        return MRR_rate / user_number

    def evolution(self):
        pre_top_k_10 = []
        pre_top_k_20 = []
        pre_top_k_50 = []

        for _ in self.test_users:
            batch_user, batch_item, batch_session, batch_pre_session = self.dg.gen_test_batch_data(
                self.batch_size)
            top_index_10, top_index_20, top_index_50 = self.sess.run(
                [self.top_index_10, self.top_index_20, self.top_index_50],
                feed_dict={self.user_id: batch_user,
                           self.item_id: batch_item,
                           self.current_session: batch_session,
                           self.pre_sessions: batch_pre_session})
            pre_top_k_10.append(top_index_10)
            pre_top_k_20.append(top_index_20)
            pre_top_k_50.append(top_index_50)

        P_10 = self.P_k(pre_top_k_10, self.test_real_items)
        MRR_10 = self.MRR_k(pre_top_k_10, self.test_real_items)

        P_20 = self.P_k(pre_top_k_20, self.test_real_items)
        MRR_20 = self.MRR_k(pre_top_k_20, self.test_real_items)

        P_50 = self.P_k(pre_top_k_50, self.test_real_items)
        MRR_50 = self.MRR_k(pre_top_k_50, self.test_real_items)

        self.logger.info(self.input_data_type + ',' + 'P@10' + ' = ' + str(P_10))
        self.logger.info(self.input_data_type + ',' + 'MRR@10' + ' = ' + str(MRR_10) + '\n')

        self.logger.info(self.input_data_type + ',' + 'P@20' + ' = ' + str(P_20))
        self.logger.info(self.input_data_type + ',' + 'MRR@20' + ' = ' + str(MRR_20) + '\n')

        self.logger.info(self.input_data_type + ',' + 'P@50' + ' = ' + str(P_50))
        self.logger.info(self.input_data_type + ',' + 'MRR@50' + ' = ' + str(MRR_50) + '\n')

        return


if __name__ == '__main__':

    # specify type of data set, types of data sets available
    type = ['tallM', 'gowalla', 'lastFM', 'fineFoods', 'movieLens', 'tafeng']
    index = 0
    print('use dataset ',type[index])

    # parameters
    # number of negative items for training, per session
    neg_number = 10
    #
    itera = 100
    # size of item and user embeddings
    global_dimension = 50

    #
    model = shan(type[index], neg_number, itera, global_dimension)

    model.build_model()

    model.run()
