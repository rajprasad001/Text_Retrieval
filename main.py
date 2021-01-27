import argparse
import pandas as pd
from similarity_check import Similarity_Metric
from data_util.data_loader import Data_load
from data_util.data_vectorizer import Data_Preprocessing
from data_util.data_vectorizer import Vectorization
from data_util.data_vectorizer import User_Input_Vectorizer


def argparser():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--dataset_path', help='data_directory', metavar='Path',
                        default='D:/content_matching/data_util/data/ArticleDataset.json')
    parser.add_argument('--output_path', help='output_path', metavar='Path',
                        default='D:/content_matching/data_util/data/')
    # parser.add_argument('--search', help='Type desired content', type = str, required=True)
    args = parser.parse_args()
    return args


def drop_columns(data_frame, column_name):
    return data_frame.drop(columns=column_name, axis=1)


def data_pre_vectotrization(data):
    print('Creating Processed File...')
    prepare_data = Data_Preprocessing(data)
    data_punc_rem = prepare_data.punctuation_removal()

    prepare_data = Data_Preprocessing(data_punc_rem)
    data_lemmatized = prepare_data.lemmatizing()

    return data_lemmatized


def embedding_data(data):
    print("Embedding data.....")
    vectorizer = Vectorization(data)
    tfidf_vectorizer, doc_matrix = vectorizer.vec_gen()
    temp_df = pd.DataFrame(doc_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())
    print('   Embedding done.')
    return tfidf_vectorizer, temp_df


def transform_user_input(input_text, tf_idf_vectorizer):
    user_data_processed = User_Input_Vectorizer(input_text, tf_idf_vectorizer)
    user_data_processed = user_data_processed.input_preprocessing()
    return user_data_processed


def similarity_score(data1, data2):
    print('Calculating Cosine similarity score.....')
    score_df = Similarity_Metric(data1, data2)
    score_df = score_df.similarity_calc()
    print('   Done.')
    return score_df


def main():
    args = argparser()

    raw_df = Data_load(args)
    raw_df = raw_df.json_to_dataframe()

    raw_df2 = drop_columns(raw_df, ['date', 'unknown'])
    processed_data = data_pre_vectotrization(raw_df2)

    tfidf_vectorizer, vector_embedding = embedding_data(processed_data)

    # user_input = args.search
    user_input = 'quantum computing software from IBM.'
    vectorized_user_input = transform_user_input(user_input, tfidf_vectorizer)

    similarity_score_list = similarity_score(vector_embedding, vectorized_user_input)
    raw_df2['similarity_score'] = similarity_score_list

    recomendation_df = raw_df2.sort_values('similarity_score', ascending=False)
    print()
    print('Providing 100 relevant documents.')
    recommended_docs = recomendation_df.head(100)
    print(recommended_docs)


if __name__ == '__main__':
    main()
