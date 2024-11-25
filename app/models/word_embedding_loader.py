import os
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

__version__ = "0.1.0"

class FastTextSimilarity:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, f"word_embedding_{__version__}.bin")
        self.model = None

    def load_model(self):
        try:
            self.model = fasttext.load_model(self.model_path)
        except Exception as e:
            raise ValueError(f"Failed to load FastText model: {str(e)}")

    def find_most_similar(self, target_word, word_list, threshold=0.4):
        """
        Find the most similar word to the target word from a list of words.

        :param target_word: The word for which similarity is calculated.
        :param word_list: List of words to compare.
        :return: Most similar word and its similarity score.
        """
        tokens = target_word.split(' ')

        most_similar_word = None
        highest_similarity = -1

        word_vectors = [self.model.get_word_vector(word) for word in word_list]

        for token in tokens[0]:
            target_vector = self.model.get_word_vector(token)

            # Calculate cosine similarities
            similarities = cosine_similarity([target_vector], word_vectors)[0]

            # Find the word with the highest similarity for this token
            max_index = np.argmax(similarities)
            token_most_similar_word = word_list[max_index]
            token_highest_similarity = similarities[max_index]

            # Update the most similar word and highest similarity if this token's similarity is higher
            if token_highest_similarity > highest_similarity:
                most_similar_word = token_most_similar_word
                highest_similarity = token_highest_similarity

        # if highest_similarity > threshold:
        #     return most_similar_word, highest_similarity
        # else:
        #     return None, 0.0
        
        return most_similar_word, highest_similarity
