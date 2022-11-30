import unittest
import pandas as pd
from unittest.mock import MagicMock

from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_train_samples = MagicMock(return_value=80)

        self.assertEqual(base._get_num_train_batches(), 4)

    def test__get_num_test_batches(self):
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        base._get_num_train_samples = MagicMock(return_value=80)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_test_batches(), 20)

    def test_get_index_to_label_map(self):
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_label_list = MagicMock(return_value=["tag1", "tag2", "tag3"])
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        expected = {
            "0": "tag1",
            "1": "tag2",
            "2": "tag3"
        }
        self.assertEqual(base.get_index_to_label_map(), expected)

    def test_index_to_label_and_label_to_index_are_identity(self):
        # TODO: CODE HERE
        pass

    def test_to_indexes(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        labels = ["tag2", "tag1", "tag3"]
        base.get_label_to_index_map = MagicMock(return_value={
            "0": "tag1",
            "1": "tag2",
            "2": "tag3"
        }
        )
        expected = [1, 0, 2]
        self.assertEqual(base.to_indexes(labels), expected)


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }
        )
        )
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        }
        )
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples_is_correct(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }
        )
        )
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        expected = 1
        self.assertEqual(dataset._get_num_samples, expected)

    def test_get_train_batch_returns_expected_shape(self):
        base = utils.Local(20, 0.8)

    def test_get_test_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        pass

    def test_get_train_batch_raises_assertion_error(self):
        # TODO: CODE HERE
        pass
