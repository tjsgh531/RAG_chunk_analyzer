import gensim
from gensim import corpora
from gensim.models import LdaModel
from konlpy.tag import Okt
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class TopicCoherence:
    def __init__(self, df):
        self.df = df
        self.okt = Okt()

        self.dictionary = None
        self.corpus = None
        self.model = None

    def text_preprocess(self, text):
        morphemes = okt.pos(text)
        result = [word for word, pos in morphemes if pos in ['Noun', 'Adjective', 'Verb']]
        return result

    def get_dictionary(self, selected_col):
        if self.dictionary is None:
            raw_documents = self.df[selected_col].to_list()

            documents = [self.text_preprocess(doc) for doc in row_documents]
            self.dictionary = corpora.Dictionary(documents)

        return self.dictionary

    def get_corpus(self):
        if self.dictionary is None:
            raise Exception("딕셔너리 안 만들어짐")

        if self.corpus is None :
            self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]

        return self.corpus

    def get_model(self, n=400):
        if self.corpus is None or self.dictionary is None:
            raise Exception("딕셔너리, 코퍼스 확인 - None")

        if self.model is None:
            self.model = LdaModel(
                corpus = self.corpus,
                id2word = self.dictionary,
                random_state = 42,
                update_every = 1,
                chunk_size = 100,
                passes = 10,
                alpha = 'auto',
                per_word_toipics = True,
            )

        return self.model

    def calculate_entropy(self, topic_distributions):
        probabilites = [prob for topic_id, prob in topic_distributions if prob > 0]
        entropy = -sum(p * math.log2(p) for p in probabilites)
        return entropy

    def get_entropy_score(self, selected_col):
        # 딕셔너리 만들기
        self.get_dictionary(selected_col)

        # corpus 만들기
        self.get_corpus()

        # model 만들기
        self.get_model()

        entropies = []
        for doc_corp in self.corpus:
            doc_topic_distributes = self.model.get_document_topics(doc_corp)
            sorted_topics = sorted(doc_topic_distributes, key = lambda x: x[1], reverse=True)
            entropy = self.calculate_entropy(sorted_topics)
            entropies.append(entropy)

        self.df['entropy_score'] = entropies

        return self.df
    
    def visualize_entropy(self):
        entropy_mean = np.mean(self.df['entropy_score'])

        fig = plt.figure(figsize = (10, 5))
        sns.histplob(self.df['entropy_score'], kde=True, bins=30)
        plt.axvline(x = 4.2, color='red', linestyle='--')
        plt.title('Entropy Distribution')
        plt.xlabel('Entropy Score')
        plt.ylabel('Frequency')

        st.pyplot(fig)
        