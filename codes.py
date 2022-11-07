#Kullanacağımız kütüphaneleri içeriye alalım.
#Veri Kazıma
from bs4 import BeautifulSoup 
import requests 

yorum=[]
for i in range(50):
  reuq = requests.get("https://www.hepsiburada.com/ara?q=kad%C4%B1n+ayakkab%C4%B1"+ "&sayfa=i")
  soup=BeautifulSoup(reuq.content, "lxml")
  s1=soup.find("div", attrs= {"class","productListContent-pXUkO4iHa51o_17CBibU"})
  ls=[]
  for i in s1.find_all("a"):
      a=i.get("href")
      ls.append(a)

  for i in ls:
    s2=requests.get("https://www.hepsiburada.com"+i)
    ayakkabi=BeautifulSoup(s2.content,"lxml")
    for comment in ayakkabi.find_all("span",attrs= {"itemprop":"description"}):
      yorum.append(comment.text)



yildiz=[]
for i in range(50):
  reuq = requests.get("https://www.hepsiburada.com/ara?q=kad%C4%B1n+ayakkab%C4%B1"+ "&sayfa=i")
  soup=BeautifulSoup(reuq.content, "lxml")
  s1=soup.find("div", attrs= {"class","productListContent-pXUkO4iHa51o_17CBibU"})
  ls=[]
  for i in s1.find_all("a"):
      a=i.get("href")
      ls.append(a)

  for i in ls:
    s2=requests.get("https://www.hepsiburada.com"+i)
    ayakkabi=BeautifulSoup(s2.content,"lxml")
    for star in ayakkabi.find_all("div", attrs= {"class":"hermes-RatingPointer-module-UefD0t2XvgGWsKdLkNoX"}):
      array=[]
      for i in star.find_all("div", class_="star"):
        array.append(i)
        st=len(array)
      yildiz.append(st)


import pandas as pd
import numpy as np

df=pd.DataFrame(yorum)
df.columns=["Yorum"]
df["yildiz"]=pd.Series(yildiz)




#Eksik değer tablosu
def eksik_deger_tb(df):
  eksik_deger=df.isnull().sum()
  eksik_deger_yuzde=100*eksik_deger/len(df)
  eksik_deger_tb=pd.concat([eksik_deger,eksik_deger_yuzde],axis=1)
  eksik_deger_tablo_son=eksik_deger_tb.rename(columns={0:"Eksik Değerler",1:"% Değeri"})
  return eksik_deger_tablo_son 

eksik_deger_tb(df)


#noktalama işaretleri kaldırılması
import string
def remove_punctuation(text):
    no_punc = [words for words in text if words not in string.punctuation]
    word_wo_punc = "".join(no_punc)
    return word_wo_punc

df["Yorum"] = df["Yorum"].apply(lambda x: remove_punctuation(x))
df["Yorum"] = df["Yorum"].apply(lambda x: x.replace("\r", " "))
df["Yorum"] = df["Yorum"].apply(lambda x: x.replace("\n", " "))



#Sayıların kaldırılması
def remove_numeric(corpus):
    output = "".join(words for words in corpus if not words.isdigit())
    return output

df["Yorum"] = df["Yorum"].apply(lambda x: remove_numeric(x)) 
df.head()



#Yorum içerisindeki tüm karakteleri küçük harfe dönüştürüyorum.
df["Yorum"]=df["Yorum"].apply(lambda x: x.lower())
df.head()


#Görselleştirme
from IPython.display import display,HTML,IFrame
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import base64
%matplotlib inline
import plotly.express as px
sns.set()
!pip install textblob
from textblob import TextBlob

df["harf_uzunluğu"]=df["Yorum"].astype(str).apply(len)   #kelime sayısı ve metin uzunluğu
df["kelime_sayisi"]=df["Yorum"].apply(lambda x:len(str(x).split()



fig=px.histogram(df,x="kelime_sayisi",nbins=200,title="Kelime Sayısı")
fig.show()

fig = px.histogram(df, x="harf_uzunlugu", nbins=20, title='Harf Uuznluğu')
fig.show()




#NLP ve Modelleme
from collections import Counter
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop=stopwords.words("turkish")

#kelimelerin frekansı Stopwordleir kaldırma işlemi
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['Yorum'], 25)
df1 = pd.DataFrame(common_words, columns = ['kelime' , 'geçiş frekansı'])
fig = px.bar(df1, x='kelime', y='geçiş frekansı',
             hover_data=['kelime', 'geçiş frekansı'], color='geçiş frekansı',
             title='Stopwordsleri kaldırmadan en çok geçen 25 kelime',
             height=400)
fig.show()



#kelimelerin frekansı Stopwordleir kaldırma işlemi
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['Yorum'], 25)
df1 = pd.DataFrame(common_words, columns = ['kelime' , 'geçiş frekansı'])
fig = px.bar(df1, x='kelime', y='geçiş frekansı',
             hover_data=['kelime', 'geçiş frekansı'], color='geçiş frekansı',
             title='Stopwordsleri kaldırmadan en çok geçen 25 kelime',
             height=400)
fig.show()



#yorumlarının kullandığı model
target = df["Yıldız"].values.tolist()
data1 = df["Yorum"].values.tolist()


import nltk
import gensim
nltk.download('stopwords')
nltk.download('punkt')

# Python program to generate word vectors using Word2Vec 
# importing all necessary modules 
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings 
warnings.filterwarnings(action = 'ignore') 
import gensim 
from gensim.models import Word2Vec 
 
stopWords = set(stopwords.words('turkish'))

sentence= """hafif güzel bir ayakkabıkızim çok sevdiyalniz mağaza öyle bir kutuya koymuşki paketi açınca kutu parçalandıayakkabiyi kutuda muhafaza etmenize imkan yok kutu çöpmagaza 
buna dikkat etmeli normal bir kutuya koymalıparamparça kutuya değil çift aldım farkli renklerde ikisininde kutusu aynı çöp yani"""
data = [] 

# iterate through each sentence in the file 
for w in sent_tokenize(sentence):
    temp = [] 
    
    # tokenize the sentence into words 
words = word_tokenize(sentence) 
for w in words:
    if w not in stopWords:
        temp.append(w)
data.append(temp)


import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud 
text = []
for i in sentence:
    text.append(i)
text = ''.join(map(str, text)) 
wordcloud = WordCloud(width=6000, height=1000, max_font_size=300,background_color='white').generate(text)
plt.figure(figsize=(20,17))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#  CBOW model 
model1 = gensim.models.Word2Vec(data, size=2, iter= 20,min_count = 1, window=5)
#Sonuçlar
print("Cosine similarity between 'kutuda' " +
            "ve  'muhafaza' - CBOW : ", 
    model1.similarity('kutuda','muhafaza')) 

print("Cosine similarity between 'kutuda' " +
            "ve  'etmenize' - CBOW : ", 
    model1.similarity('kutuda','etmenize'))


# Skip Gram model 
model2 = gensim.models.Word2Vec(data, min_count=1, size = 2,iter=20, window = 10) 
#Sonuçlar
print("Cosine similarity between 'kutuda' " +
            "ve  'muhafaza' - Skip Gram : ", 
    model2.similarity('kutuda','muhafaza')) 
    
print("Cosine similarity between 'kutuda' " +
            "ve 'etmenize' - Skip Gram : ", 
    model2.similarity('kutuda','etmenize')