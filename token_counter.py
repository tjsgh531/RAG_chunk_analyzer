from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

class TokenCounter:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')

    def count_tokens(self, text):
        return len(self.tokenizer.tokenize(text))

    def token_count(self, df, col):
        df['token_num'] = df[col].apply(self.count_tokens)
        return df

    def visualize_token(self, df):
        token_mean = np.mean(df['token_num'])
        toekn_var = np.var(df['token_num'])

        fig = plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        sns.histplot(df['token_num'], kde=True, bins=30)
        plt.title('Token Count Distribution')
        plt.xlabel('Token Count')
        plt.ylabel('Frequency')

        plt.subplot(2, 1, 2)
        sns.boxplot(x=df['token_num'])
        plt.title('Token Count Boxplot')
        plt.xlabel('Token Count')
        
        st.write(f"토큰 수 평균 : {token_mean:.4f}")
        st.pyplot(fig)
