#!/usr/bin/env python
# coding: utf-8

# # 🔍 참 언론인을 찾아라! 기자 추천하기

# "
# __한국 언론__?
# "
# 
# [__40개국 중 40위…한국 언론 신뢰도 4년째 최하위__](https://www.seoul.co.kr/news/newsView.php?id=20200617500048)
# 
# ```
# 영국 옥스퍼드대학교 부설 로이터저널리즘연구소가 최근 공개한 ‘디지털뉴스리포트 2020’에 따르면 
# 한국인들의 뉴스 신뢰도는 21%로 조사 대상 40개국 중 40위로 나타났다.😭😭😭
# ```
# 
# ---
# 
# "
# __저널리즘을 빛내는, 좋은 글을 쓰는 기자가 1명은 있지 않을까요?__
# "
# ```
# 분야별 최고의 기자는? 신문사별 최고의 기자는? 내 취향에 맞는 글을 쓰는 기자는?....
# 
# ```
# 
# ---
# 
# 네이버 기자 프로필 데이터 기반하여 참 언론인, 기자 추천하기 프로젝트!

# ## TEAM 😀
# 
# - PM(Project Manager) 1명: 프로젝트의 총담당자입니다. 전체 일정을 관리합니다. 발표자료를 만들고, 발표를 합니다.
# - PE(Project Engineer) 5~6명
#      - Crawling & Preprocessing: 데이터 크롤링 및 전처리를 담당합니다. 모조리 긁어모아 깔끔하게 정리합니다.
#      - EDA: 데이터 분석을 합니다. 데이터의 숨은 이야기를 찾습니다.
#      - Recommendation: 추천 알고리즘을 만듭니다. 

# ## Import Library (step.01)

# In[1]:


import pandas as pd
import pandas_profiling
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.max_rows=150
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import glob

# NLP
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# crawling
from konlpy.tag import Mecab
import requests
import pandas as pd
from bs4 import BeautifulSoup


# In[2]:


url = 'https://media.naver.com/journalist/001/74226'


# ### step.02 crawling
# 1. 반복문으로 기자 프로필 url 리스트 만들기
# 2. 기자별 구독수, 응원수, 30일 평균 작성 기사 수, 대표 섹션, 구독자 통계 수집
# 3. 주간 많이 본 뉴스 기사 추천 수, 반응 [좋아요, 훈훈해요, 슬퍼요, 화나요, 후속기사 원해요] 수집
# 
# ### step.03 EDA
# 1. 신문사별, 섹션별, 기자별 분석
# 2. 기사 내용 기준 유사한 기자
# 
# ### step.04 recommendation
# 1. 섹션별 기자 추천
# 2. 신문사별 기자 추천
# 3. 실시간 주목받는 기자 추천
# 4. 서로 유사한 글 성향의 기자 추천
# 5. 그 외 등등
