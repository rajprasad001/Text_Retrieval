from sklearn.metrics.pairwise import cosine_similarity


class Similarity_Metric:
    def __init__(self, emeddding_1, embedding_2):
        self.database_embedding = emeddding_1
        self.input_embedding = embedding_2

    def similarity_calc(self):
        simlarity_score_list = []
        database_embedding = self.database_embedding
        input_embedding = self.input_embedding
        input_matrix = input_embedding.values.tolist()
        for index, rows in database_embedding.iterrows():
            doc_matrix = rows.values.tolist()
            cosine_sim_score = cosine_similarity([doc_matrix], input_matrix)
            simlarity_score_list.append(cosine_sim_score)
        return simlarity_score_list
