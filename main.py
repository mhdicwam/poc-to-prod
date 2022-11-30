import pandas as pd
from preprocessing.preprocessing import utils
from preprocessing.preprocessing.embeddings import embed


path = "stackoverflow_posts.csv"


# base = utils.BaseTextCategorizationDataset(20, 0.8)
cat_base = utils.LocalTextCategorizationDataset(path, 20)

# print(cat_base.load_dataset(path, 100))

# print(cat_base._get_label_list())

# print(cat_base.get_train_batch())


print(cat_base.get_train_batch()[1].shape)




