import json
import pandas as pd
import streamlit as st
from huggingface_hub import login, whoami

from token_counter import TokenCounter

class ChunkAnalyzer:
    def __init__(self):
        # initialize session state
        self.session_init()

        # 데이터 입력 받기 창
        self.df = None
        self.get_data()

        if self.df is not None:
            # hugging face login
            self.login_huggingface()

            # 로그인에 성공하면
            if st.session_state.hf_login:
                self.token_count_analyzer = TokenCounter()
                self.select_columns()

                # 분석 시작
                start_btn = st.button("분석하기", key='start')
                self.start_analyze(start_btn)

    def session_init(self):
        if 'hf_login' not in st.session_state:
            st.session_state['hf_login'] = False
        
        if 'hf_token' not in st.session_state:
            st.session_state['hf_token'] = ''        


    def get_data(self):
        uploaded_file = st.file_uploader("JSONL 파일을 선택하세요", type = "jsonl")
        
        if uploaded_file is not None:
            content = uploaded_file.getvalue().decode("utf-8")
    
            # JSONL 파일 파싱
            data_list = [json.loads(line) for line in content.split('\n') if line.strip()]
            self.df = pd.DataFrame(data_list)

            st.dataframe(self.df)

    # hugging face 로그인
    def login_huggingface(self):
        hf_token = st.text_input("Hugging Face API Token을 입력하시오 : ", key="hf_token")
        st.session_state['hf_login'] = False

        if hf_token:
            login(token=hf_token)

            try:
                user_info = whoami()
                st.write("hugging face 로그인 성공!! 모델 가져오는 중 ...")
                st.session_state.hf_login = True
            except Exception as e:
                st.error("로그인 실패...")
                st.session_state.hf_token = ""
    
    def select_columns(self):
        columns = self.df.columns
        self.selected_column = st.radio("분석할 열을 선택하시오 : ", columns)

    def start_analyze(self, start_btn):
        if st.session_state.hf_login and start_btn:
            # 토큰 수 분석
            self.df = self.token_count_analyzer.token_count(self.df, self.selected_column)

            # 토큰 수 분석 시각화
            self.token_count_analyzer.visualize_token(self.df)
            

if __name__ == '__main__':
    ChunkAnalyzer()