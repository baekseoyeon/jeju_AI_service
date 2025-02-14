import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import faiss
import streamlit as st
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
import google.generativeai as genai
import time




# # 경로 설정
# data_path = './data'
# module_path = './modules'

# Gemini 설정
import google.generativeai as genai

# import shutil
# os.makedirs("/root/.streamlit", exist_ok=True)
# shutil.copy("secrets.toml", "/root/.streamlit/secrets.toml")

# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
# genai.configure(api_key=GOOGLE_API_KEY)
genai.configure(api_key="key")  # API 키를 직접 입력
model = genai.GenerativeModel("gemini-1.5-flash")

# 파일 경로 설정
faiss_index_path = "9000_faiss_index_path"
document_ids_path = "9000_chunk_document_ids_path"
file_path = '9000_chunk_sentence_modify_v18_result_final.csv'
user_dictionary_path = 'user_dictionary.txt'  # 사용자 사전 파일 경로

# # CSV 파일 로드
# csv_file_path = os.path.join(data_path, "jeju_final_sentences_v7_modify.csv")
# df = pd.read_csv(csv_file_path)

####------------UI-------------###
# Streamlit App UI
st.set_page_config(layout="wide", page_title="🍽️좀 색다른 맛집을 찾는다구? 우리가 해결해줄게!🍽️")

import streamlit as st

# ------------------------전체 배경/글꼴------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Jua&display=swap');
    * {
        font-family: 'Jua', sans-serif;
    }

    .stApp {
        background-color: #fdd8b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------타이틀------------------------
st.markdown(
    """
    <h1 style='font-family: "Jua", sans-serif;'>🍊 나만의 제주도 맛집 여행 🍊</h1>
    """,
    unsafe_allow_html=True
)

# ------------------------앱 소개------------------------
with st.expander('다들 모르는 그 곳, 궁금하지 않으세요? 👀'):
    st.markdown(
    """
    <style>
    .stExpander {
        background-color: #faded0;
        padding: 5px;  /* 패딩 추가 */
        border-radius: 10px;  /* 모서리를 둥글게 */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    """
    <h3 style='font-family: "Jua", sans-serif;'>남들은 정말 운이 좋은 걸까요? 나는 왜 매번 이렇게 긴 웨이팅에 지칠까요?</h3>
    """,
    unsafe_allow_html=True
    )
    st.write('기대감을 안고 어렵게 한 입 맛봤지만, 그저 그런 맛이었을 때... 모두 이러한 경험이 있지 않으신가요? 😢')
    st.write('저희 앱은 그런 실망스러운 경험 대신, **💎숨겨진 보석 같은 맛집💎** 을 추천해 드립니다.')
    st.write('유명한 음식점들에 가려져 있던 **제주도의 진짜 맛집**, 그곳을 저희가 알려드릴게요! 🍽️')

@st.cache_data
def load_data():
    data = pd.read_csv(file_path)
    return data

@st.cache_data
# 데이터 및 인덱스 로드
def load_data_and_index():
    if "data_loaded" not in st.session_state:
        # CSV 파일 로드 및 chunk_texts 생성
        data = pd.read_csv(file_path)
        chunk_texts = []
        
        for idx, row in data.iterrows():
            text_chunks = []
            for col in ["업종명", "주소", "기준연월", "이용건수구간", '이용금액구간', "요일별", "시간대별", "성별", "연령별", "현지인", 'url']:
                chunk_text = row[col]
                if pd.notna(chunk_text):
                    text_chunks.append(chunk_text)
            chunk_texts.append(" ".join(text_chunks))
        st.session_state.chunk_texts = chunk_texts
        print("chunk_texts 생성 완료")
        
        # document_ids 로드
        st.session_state.chunk_document_ids = np.load(document_ids_path, allow_pickle=True).tolist()
        print("Document IDs 로드 완료")
        
        # FAISS 인덱스 로드
        st.session_state.index = faiss.read_index(faiss_index_path)
        print("FAISS 인덱스 로드 완료")
        
        # Kiwi 형태소 분석기와 사용자 사전 로드
        kiwi = Kiwi()
        added_words_count = kiwi.load_user_dictionary(user_dictionary_path)
        print(f"사용자 정의 사전에서 {added_words_count}개의 단어가 추가되었습니다.")
        st.session_state.kiwi = kiwi
        
        # BM25에 사용할 코퍼스 토큰화 및 BM25 모델 생성
        tokenized_corpus = [[token.form for token in st.session_state.kiwi.tokenize(doc)] for doc in st.session_state.chunk_texts]
        st.session_state.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 모델 생성 완료")

        # BGE 토크나이저 및 모델 설정 (CPU에서 실행)
        try:
            st.session_state.tokenizer_bge = AutoTokenizer.from_pretrained("upskyy/bge-m3-korean")
            st.session_state.model_bge = AutoModel.from_pretrained("upskyy/bge-m3-korean")  # .cuda() 제거
            print("BGE 토크나이저 및 모델 설정 완료")
        except Exception as e:
            print(f"토크나이저 로드 중 오류 발생: {e}")
            # 예외 처리 로직 추가
        
        st.session_state.data_loaded = True  # 데이터가 로드된 상태로 플래그 변경
    else:
        print("데이터 및 인덱스가 이미 로드되었습니다.")

# BM25 + FAISS 병렬 검색 함수 (세션 상태에 저장된 데이터 활용)
def search_documents(query, bm25_k=1000, faiss_k=10):
    # 1. Kiwi를 이용한 BM25 1차 검색
    tokenized_query_kiwi = [token.form for token in st.session_state.kiwi.tokenize(query)]
    bm25_results = st.session_state.bm25.get_top_n(tokenized_query_kiwi, st.session_state.chunk_texts, n=bm25_k)
    
    # 2. BM25 필터링된 문서들의 document_id 추출
    bm25_filtered_ids = [st.session_state.chunk_document_ids[st.session_state.chunk_texts.index(doc)] for doc in bm25_results]
    
    # 3. BGE 토크나이저로 쿼리 임베딩 생성 (CPU에서 실행)
    query_inputs_bge = st.session_state.tokenizer_bge(query, return_tensors="pt", padding=True, truncation=True)  # .to('cuda') 제거
    with torch.no_grad():
        query_outputs_bge = st.session_state.model_bge(**query_inputs_bge)
        query_embedding_bge = query_outputs_bge.last_hidden_state.mean(dim=1).cpu().numpy()  # GPU 관련 코드 제거
    
    # Normalize query embedding for cosine similarity
    faiss.normalize_L2(query_embedding_bge)
    
    # FAISS 검색 수행
    distances, indices = st.session_state.index.search(query_embedding_bge, faiss_k)
    print(f"FAISS 검색 완료 - 상위 {faiss_k}개의 검색 결과:")
    
    # FAISS에서 찾은 상위 문서들의 document_id 추출
    retrieved_document_ids = [st.session_state.chunk_document_ids[idx] for idx in indices[0]]
    unique_document_ids = list(dict.fromkeys(retrieved_document_ids))  # 중복 제거
    
    # 최종 검색된 문서들 반환
    retrieved_documents = [st.session_state.chunk_texts[doc_id] for doc_id in unique_document_ids]
    
    # 검색된 문서 출력
    for i, doc in enumerate(retrieved_documents, 1):
        print(f"Document {i}: {doc}")

    return retrieved_documents

# LLM(Gemini) 응답 생성 함수
def generate_gemini_response(query, retrieved_documents):
    # 검색된 문서들을 프롬프트로 결합
    context = " ".join(retrieved_documents)
    
    # 모델에 전달할 최종 프롬프트 구성
    prompt = (
        f"As a assistant helping with tourism in Jeju Island, "
        f"please refer to the following information:\n{context}\n\n"
        f"Based on this information, please provide a response to the following query:\n'{query}'"
    )
    
    # Gemini 모델 초기화 및 응답 생성
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_model.generate_content(prompt)
    return response.text

# 세션 상태 초기화 및 데이터 로드 (최초 실행 시 데이터 로드)
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 데이터 로드 함수 호출 (최초 실행 시에만)
load_data_and_index()

# 사용자 입력 쿼리문 (st.text_input으로 수정)
prompt = st.text_input("위치와 음식 종류를 입력해보세요", "")

# 사용자가 입력한 메시지 처리
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    # st.write(f"User: {prompt}")

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.spinner("맛집을 찾고 있어요! 🍜🍲🍗🍣🥯"):
            # 1. BM25 + FAISS 병렬 검색 수행
            retrieved_documents = search_documents(prompt)
            
            # 2. 검색된 문서를 바탕으로 LLM(Gemini) 응답 생성
            full_response = generate_gemini_response(prompt, retrieved_documents)

            placeholder = st.empty()
            placeholder.markdown(full_response)
        
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
