#!/usr/bin/env python
# coding: utf-8

# #  📈 미래의 가치를 예측하다, 주식 종목 추천

# 주식 열풍이다. 
# 
# [주식에 빠진 대한민국.. 2020년 직장인 수익률 56%](https://www.fnnews.com/news/202101181000209402)
# 
# ["주식 안 하면 바보 된다"…전세금까지 베팅하는 개미들](https://www.hankyung.com/finance/article/2021011106321)
# 
# 파이썬으로 주식을 해보자. 어떤 종목을 사야할까요?

# ## TEAM 😀
# 
# - PM(Project Manager) 1명: 프로젝트의 총담당자입니다. 전체 일정을 관리합니다. 발표자료를 만들고, 발표를 합니다.
# - PE(Project Engineer) 5~6명
#      - Crawling & Preprocessing: 데이터 크롤링 및 전처리를 담당합니다. 모조리 긁어모아 깔끔하게 정리합니다.
#      - EDA: 데이터 분석을 합니다. 데이터의 숨은 이야기를 찾습니다.
#      - Recommendation: 추천 알고리즘을 만듭니다. 

# ---

# ## 주식 종목 추천 방법
# 
# #### 1) 데이터 분석 기반 미래 주식 가격 예측하여 종목 추천하기
# - 데이터 수집: 종목별 과거~현재 주식 가격 데이터   [데이터 수집 링크](https://github.com/choosunsick/Korea_Stocks)
# - 데이터 수집: 종목별 재무 상태 데이터 [데이터 수집 링크](https://opendart.fss.or.kr/intro/main.do)
# - 수집한 데이터를 시계열 데이터로 전처리하기 + EDA
# - LGBM 등의 머신러닝 알고리즘 활용하여 종목별 미래 주식 가격 예측 모델 학습
# - 추천보다는 데이터 분석 + 지도학습 = 회귀분석에 해당
# 
# #### 2) 애널리스트 투자의견 기반 종목 추천하기
# - 데이터 수집: 애널리스트 종목별 투자의견, 적정가격 데이터 수집 [데이터 수집 링크](http://consensus.hankyung.com/apps.analysis/analysis.list?&sdate=2018-01-18&edate=2021-01-18&report_type=CO&pagenum=80&order_type=&now_page=6)
# - 수집한 데이터를 애널리스트-종목 평점 행렬로 변환
# - MF, CF, CBF 등의 추천 알고리즘 활용하여 종목 추천
# - 기간에 따른 추천 유효성, 영향도 등 종합 검증 필요
# 
# #### 3) 

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

