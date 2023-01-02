import pandas as pd
import numpy as np
from preprocessing.preprocessing import utils
from preprocessing.preprocessing.embeddings import embed
import os
import unittest
from preprocessing.preprocessing.utils import _SimpleSequence
from predict.predict.run import *

path = "stackoverflow_posts.csv"

base = utils.BaseTextCategorizationDataset(20, 0.8)
cat_base = utils.LocalTextCategorizationDataset(path, 100)

# print(cat_base.get_train_sequence().get_batch_method)
# print((cat_base.get_train_sequence()))

# print(cat_base._get_num_samples())
#
# print(base._get_num_test_batches())
# cat_base = utils.BaseTextCategorizationDataset
# print(cat_base.load_dataset(path, 100))

# df = cat_base.load_dataset(path, min_samples_per_label=100)
#

predict = TextPredictionModel()

# print(df.head())
# print(cat_base._get_label_list())
# #
# # # print(cat_base.get_train_batch())
# # cwd = os.getcwd()
# # print(os)
# # print(cat_base.get_n)
# print(cat_base.get_num_labels())
#
# print(cat_base.get_train_sequence())
# print(type(cat_base.get_test_batch()[0]))
# print(type(cat_base.get_test_batch()[1]))
#
# print(len(cat_base.get_test_batch()[0]))
#
# print((cat_base.get_test_batch()[1]).shape)
#
# print(len(cat_base.get_test_batch()[1]))
# print(cat_base.get_label_to_index_map())
