# cognitive_load_estimator_gugudolls
adaptive cognitive load estimator

#important commands
-> streamlit run app.py
-> source venv/bin/activate
-> ctrl+c to terminate venv

#important modules to be imported
import streamlit as st

import re
import pandas as pd
import spacy
import textstat as tx
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

scikit-learn : { from sklearn.feature_extraction.text import TfidfVectorizer
                 from sklearn.decomposition import LatentDirichletAllocation
                 from sklearn.cluster import KMeans }

data cleaning:  { from sentence_transformers import SentenceTransformer
                  from nltk.corpus import stopwords
                  import nltk }

import pytextrank #for topic modelling
