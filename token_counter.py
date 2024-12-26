from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

class TokenCounter:
    def __init__(self, df):
        self.tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')
        self.set_df(df)

    def count_tokens(self, text):
        return len(self.tokenizer.tokenize(text))

    def set_df(self, df):
        self.df = df

    def token_count(self, col):
        self.df['token_num'] = self.df[col].apply(self.count_tokens)
        return self.df

    def visualize_token(self):
        token_mean = np.mean(self.df['token_num'])

        fig = plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        sns.histplot(self.df['token_num'], kde=True, bins=30)
        plt.title('Token Count Distribution')
        plt.xlabel('Token Count')
        plt.ylabel('Frequency')

        plt.subplot(2, 1, 2)
        sns.boxplot(x=self.df['token_num'])
        plt.title('Token Count Boxplot')
        plt.xlabel('Token Count')
        
        st.write(f"토큰 수 평균 : {token_mean:.4f}")
        st.pyplot(fig)
