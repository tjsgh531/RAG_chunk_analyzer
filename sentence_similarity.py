from sentence_transformers import SentenceTransformer, util
import torch
import re
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

class SentenceSimilarity:
    def __init__(self,df):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS').to(self.device)
        self.df = df
        print("device: ", self.device)

    def split_sentence(self, text):
        sentences = re.split(r'[.!?]', text)
        return [sent.strip() for sent in sentences if sent.strip() != '']

    def get_sent_similarity(self, selected_column):
        text_list = self.df[selected_column].to_list()
        sentences = [self.split_sentence(text) for text in text_list]

        results = []
        for sent in sentences:
            embeddings = self.model.encode(sent, convert_to_tensor=True).to(self.device)
            sent_length = len(sent)

            sent_sim_scores = []
            for i in range(sent_length):
                for j in range(i+1, sent_length):
                    sent_1 = embeddings[i]
                    sent_2 = embeddings[j]
                    sim_score = util.pytorch_cos_sim(sent_1, sent_2)
                    sent_sim_scores.append(sim_score.item())

            sim_score = np.mean(sent_sim_scores)
            results.append(sim_score)

        self.df["sim_score"] = results
        return self.df

    def visualize_sim(self):
        sim_mean = np.mean(self.df['sim_score'])

        fig = plt.figure(figsize=(10, 5))
        sns.histplot(self.df['sim_score'], kde = True, bins=30)
        plt.axvline(x=0.3, color='red', linestyle='--')
        plt.title("Sentence Similarity Score")
        plt.xlabel("text data")
        plt.ylabel("Frequency")

        st.pyplot(fig)

