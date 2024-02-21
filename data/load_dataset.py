import pandas as pd
import os
import requests

class LoadDataset:
    PATH_DATASET = 'resources/dataset'
    URL_DATASET_WMT14 = 'https://huggingface.co/api/datasets/wmt14/parquet/fr-en/train/0.parquet'
    NAME_DATASET_WMT14 = 'WMT14'

    def __init__(self):
        pass

    def importWMT14(self):
        if not os.path.exists(self.PATH_DATASET):
            os.makedirs(self.PATH_DATASET)

        path_file_WMT14 = self.PATH_DATASET + '/' + self.NAME_DATASET_WMT14 + '.parquet'
        if not os.path.isfile(path_file_WMT14):
            response = requests.get(self.URL_DATASET_WMT14)

            with open(path_file_WMT14, 'wb') as f:
                f.write(response.content)

        dataset_WMT14 = pd.read_parquet(path_file_WMT14)

        dataset_en, dataset_fr = self.split_language(dataset_WMT14, 'en', 'fr')

        return dataset_en, dataset_fr

    def split_language(self, dataset, first_language, second_language):
        text_first_language = [i for i in dataset['translation.en']]
        text_second_language = [i for i in dataset['translation.fr']]
        # text_first_language = [i[first_language] for i in dataset['translation']]
        # text_second_language = [i[second_language] for i in dataset['translation']]
        return text_first_language, text_second_language