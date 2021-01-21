import json
import numpy as np
import pandas as pd


class Data_load:
    def __init__(self, path):
        self.location = path

    def json_to_dataframe(self):
        raw_data = open(self.location)
        json_dictionary = json.load(raw_data)
        df = pd.DataFrame.from_dict(json_dictionary, orient='columns')
        return df
