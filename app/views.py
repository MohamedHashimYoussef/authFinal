import pickle

from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

lemmatiser = WordNetLemmatizer()
# Create your views here.



def index(request):
    return render(request , 'index.html')

class Write(APIView):
    def post(self , request , *args , **kwargs):
        data = request.data
        tex = data['text']
        article = re.sub('[^a-zA-Z]', ' ', tex)
        article = article.lower()
        article = article.split()
        ps = PorterStemmer()
        article = [ps.stem(i) for i in article if not i in set(stopwords.words('english'))]
        article = ' '.join(article)
        lis  = []
        lis.append(article)
        n = np.array(lis)
        # vectorizer_train = CountVectorizer(analyzer=text_process)
        p = pickle.load(open("app/count", "rb"))
        bow = p.transform(n)
        xx = pickle.load(open("app/model", "rb"))
        pr = xx.predict(bow)
        print(pr)
        return HttpResponse(pr[0])