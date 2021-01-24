import json
import pandas as pd


class Data_load:
    def __init__(self, args):
        self.location = args.dataset_path

    def json_to_dataframe(self):
        raw_data = open(self.location)
        json_dictionary = json.load(raw_data)
        df = pd.DataFrame.from_dict(json_dictionary, orient='columns')
        return df
