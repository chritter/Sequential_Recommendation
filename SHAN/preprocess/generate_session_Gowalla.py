import pandas as pd
import numpy as np
import os


# used to generate datatype_dataset.csv file
# sessions

class generate(object):

    def __init__(self, dataPath, sessPath):
        self._data = pd.read_csv(dataPath,sep='\t')
        print(self._data.head())
        self.sessPath = sessPath

    def stati_data(self):
        '''
        Generate data statistics.
        :return:
        '''
        print('Total data length/entries:', len(self._data))
        print('Total number of sessions:', len(self._data.drop_duplicates(['UserId', 'Time'])))
        print('Average session length:', len(self._data) / len(self._data.drop_duplicates(['UserId', 'Time'])))
        print('Total number of users:', len(self._data.drop_duplicates('UserId')))
        print('Average number of sessions per user:',
              len(self._data.drop_duplicates(['UserId', 'Time'])) / len(self._data.drop_duplicates('UserId')))
        print('Total item number:', len(self._data.drop_duplicates('ItemId')))
        print('Data set time span：', min(self._data.Time), '~', max(self._data.Time))

    def reform_u_i_id(self):
        '''
        Renumber the item and user in the data, and then generate the session
        :return:
        '''
        user_to_id = {}
        item_to_id = {}
        # 对user进行重新编号
        user_count = 0
        item_count = 0
        for i in range(len(self._data)):
            # 对user 和 item同时进行重新编号
            u_id = self._data.at[i, 'UserId']
            i_id = self._data.at[i, 'ItemId']
            if u_id in user_to_id.keys():
                self._data.at[i, 'UserId'] = user_to_id[u_id]
            else:
                user_to_id[u_id] = user_count
                self._data.at[i, 'UserId'] = user_count
                user_count += 1
            if i_id in item_to_id.keys():
                self._data.at[i, 'ItemId'] = item_to_id[i_id]
            else:
                item_to_id[i_id] = item_count
                self._data.at[i, 'ItemId'] = item_count
                item_count += 1

        self._data.to_csv('../data/middle_data.csv', index=False)

        print('user_count', user_count)
        print('item_count', item_count)

    # According to the experimental design, the test session is randomly selected from the last month of the data set
    # to extract 20% of the session. The test set used in TallM is the last session of each user.
    def generate_train_test_session(self):

        print('statistics ... ')
        self.stati_data()  # 统计数据集


        print('encode ... ')
        self.reform_u_i_id()  # Recode user and items to get user and item indices


        print('generate train and test session ... ')
        self._data = pd.read_csv('../data/middle_data.csv')
        # split into train and test set according to time
        self._train_data = self._data[self._data.Time < 20100931].reset_index(drop=True)
        self._test_data = self._data[self._data.Time > 20100930].reset_index(drop=True)
        os.remove('../data/middle_data.csv')

        # write out if exist
        session_train_path = self.sessPath + '_train_dataset.csv'
        print('write train data of shape ',self._train_data.shape,' into ',session_train_path)
        session_test_path = self.sessPath + '_test_dataset.csv'
        print('write test data of shape ',self.session_test_path.shape,' into ',session_test_path)

        if os.path.exists(session_train_path):
            os.remove(session_train_path)

        if os.path.exists(session_test_path):
            os.remove(session_test_path)

        # for train session
        # To consider the last session, the current session is not considered in the current session.
        with open(session_train_path, 'a') as session_train_file:
            user_num = len(self._data['UserId'].drop_duplicates())
            # users = len(self._train_data['UserId'].drop_duplicates())
            item_num = len(self._data['ItemId'].drop_duplicates())
            session_train_file.write(str(user_num) + ',' + str(item_num) + '\n')
            last_userid = self._train_data.at[0, 'UserId']
            last_time = self._train_data.at[0, 'Time']
            session = str(last_userid) + ',' + str(self._train_data.at[0, 'ItemId'])
            for i in range(1, len(self._train_data)):
                # 文件使用降序打开
                # 最终session的格式为user_id,item_id:item_id...@item_id:item_id...@...
                userid = self._train_data.at[i, 'UserId']
                ItemId= self._train_data.at[i, 'ItemId']
                time = self._train_data.at[i, 'Time']
                if userid == last_userid and time == last_time:
                    # 需要将session写入到文件中，然后开始
                    session += ":" + str(itemid)
                elif userid != last_userid:
                    session_train_file.write(session + '\n')
                    last_userid = userid
                    last_time = time
                    session = str(userid) + ',' + str(itemid)
                else:
                    session += '@' + str(itemid)
                    last_time = time

        # for test session
        # To consider the last session, the last session is not considered in the current loop
        # First build session, one session line, then randomly select 20%

        with open(session_test_path, 'a') as session_test_file:
            last_userid = self._test_data.at[0, 'UserId']
            last_time = self._test_data.at[0, 'Time']
            session = str(last_userid) + ',' + str(self._test_data.at[0, 'ItemId'])
            for i in range(1, len(self._test_data)):
                # 最终session的格式为user_id,item_id:item_id...
                userid = self._test_data.at[i, 'UserId']
                ItemId= self._test_data.at[i, 'ItemId']
                time = self._test_data.at[i, 'Time']
                if userid == last_userid and time == last_time:
                    # 需要将session写入到文件中，然后开始
                    session += ":" + str(itemid)
                elif userid != last_userid:
                    session_test_file.write(session + '\n')
                    last_userid = userid
                    last_time = time
                    session = str(userid) + ',' + str(itemid)
                else:
                    session_test_file.write(session + '\n')
                    last_time = time
                    session = str(userid) + ',' + str(itemid)


if __name__ == '__main__':
    datatype = ['tallM', 'gowalla']

    #dataPath = '../data/' + datatype[1] + '_data.csv'
    dataPath = '/Users/christian/StatCan/Projects/RecommendationSystems/RecomSystemLibrary/RecommenderSystems/sequentialRec/data/gowalla/gowalla_train_tr.txt'
    sessPath = '../data/' + datatype[1]

    # read in file, into object
    object = generate(dataPath, sessPath)
    # object.stati_data()

    # create file statistics and split file into training and test set
    # write train set into
    object.generate_train_test_session()
