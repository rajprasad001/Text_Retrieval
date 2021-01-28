## A NLP approach towards finding relevant documents from a Database.

- Natural Language Processing is an important technique that offers various methods to extract useful information from a textual database. It is widely used in recommendation systems, language translation etcetra.

 - The goal of this project was to extract features from a database containing around 8000 documents. Further finding most relevant documents related to the user querry.
 
 - The general pipeline of the project is given below:
   - Loading Json file into a dataframe.
   - Removing irrelevant columns.
   - Removal of Punctuations from the text.
   - Lemmatizing the dataset.
   - Removal of stopwords.
   - Lowercsing the texts.
   - Embedding the corpus using TF-IDF vectorization technique.
   - Preprocessing User Querry.
   - Vectorization of Querry.
   - Sorting top matching documents on the basis of Cosine Similarity between the document and the querry.
   
## Dependencies
clone the repository by using git clone 
