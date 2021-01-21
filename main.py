import os
import argparse
import pandas as pd
from pathlib import Path
from data_util.data_loader import Data_load
from data_util.data_vectorizer import Data_preprocessing


def argparser():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--dataset_path', help='data_directory', metavar='Path',
                        default='D:/content_search/data_util/data/ArticleDataset.json')
    parser.add_argument('--output_path', help='output_path', metavar='Path',
                        default='D:/content_matching/data_util/data/')
    args = parser.parse_args()
    return args


def drop_columns(data_frame, column_name):
    return data_frame.drop(columns=column_name, axis=1)


def data_pre_vectotrization(data):
    prepr_data = Data_preprocessing(data)
    data_lowercase = prepr_data.lower_case()
    prepr_data = Data_preprocessing(data_lowercase)
    data_punc_removed = prepr_data.punctuation_removal()
    prepr_data = Data_preprocessing(data_punc_removed)
    data_stopwords_removed = prepr_data.stopword_removal()
    return data_stopwords_removed


def main():
    args = argparser()
    raw_df = Data_load(args.dataset_path)
    raw_df = raw_df.json_to_dataframe()
    raw_df2 = drop_columns(raw_df, ['date', 'unknown'])
    filename = 'processed_file.csv'
    try:
        with open("{}/{}".format(args.output_path, filename)) as f:
            print('Processed File Available.')
    except:
        print('Creating Processed File...')
        processed_data = data_pre_vectotrization(raw_df2)
        processed_data.to_csv(("{}/{}".format(args.output_path, filename)), index=False)


if __name__ == '__main__':
    main()
