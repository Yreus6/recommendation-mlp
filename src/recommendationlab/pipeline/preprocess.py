import os
import numpy as np
import pandas as pd

from src.recommendationlab import config


class Preprocessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def process_data(self):
        user_path = os.path.join(self.data_dir, 'users_dataset.csv')
        item_path = os.path.join(self.data_dir, 'items_dataset.csv')
        interaction_path = os.path.join(self.data_dir, 'interactions_dataset.csv')

        user_df = pd.read_csv(user_path)
        item_df = pd.read_csv(item_path)

        interaction_df = pd.read_csv(interaction_path)
        data = interaction_df.sort_values(['USER_ID', 'TIMESTAMP'])
        groups = data.groupby('USER_ID')
        data['count'] = groups['USER_ID'].transform('count')
        data['rank'] = groups.cumcount() + 1

        splits = []
        prev_threshold = None
        ratios = [0.8, 0.1, 0.1]
        for threshold in np.cumsum(ratios):
            condition = data['rank'] <= round(threshold * data['count'])
            if prev_threshold is not None:
                condition &= data['rank'] > round(prev_threshold * data['count'])
            splits.append(data[condition].drop(['rank', 'count'], axis=1))
            prev_threshold = threshold

        train, val, test = splits

        val = val[val['USER_ID'].isin(train['USER_ID'].unique())]
        val = val[val['ITEM_ID'].isin(train['ITEM_ID'].unique())]
        test = test[test['USER_ID'].isin(train['USER_ID'].unique())]
        test = test[test['ITEM_ID'].isin(train['ITEM_ID'].unique())]
        test = test[test['USER_ID'].isin(val['USER_ID'].unique())]
        test = test[test['ITEM_ID'].isin(val['ITEM_ID'].unique())]

        val = val.groupby('USER_ID').last().reset_index()
        test = test.groupby('USER_ID').last().reset_index()

        train.to_csv(os.path.join(config.SPLITSPATH, 'train.csv'), index=False)
        val.to_csv(os.path.join(config.SPLITSPATH, 'val.csv'), index=False)
        test.to_csv(os.path.join(config.SPLITSPATH, 'test.csv'), index=False)

        print('Processed data saved in {}'.format(config.SPLITSPATH))
