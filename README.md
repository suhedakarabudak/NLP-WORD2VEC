# NLP-WORD2VEC
 Doğal Dİl İşleme
 
**Python ile [Hepsiburada.com](http://Hepsiburada.com) ‘dan Verilerin Kazınması**

Projeden bahsetmek gerekirse hepsiburada da olan kadın ayakkabılarına yapılan yorumlar ve o yorumlara gelen yıldızları çekme işlemi ve Nlp modellerinden 
word2vec modelini kullanarak projeyi geliştirmek.

Bu çalışmada  web scraping yapmadan önce Python’un  [BeautifulSoup](https://pypi.org/project/beautifulsoup4/) adlı kütüphanesini kullanacağımı belirtmek isterim.Eğer daha fazla bilgi almak istersiniz [https://www.crummy.com/software/BeautifulSoup/bs4/doc/](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) inceleyebilirsiniz.

##Web scraping yaptıktan dataframe oluşturarak modelimizde kullanacağımız veri setini elde ediyorum.
|  | Yorum | Yıldız |
| --- | --- | --- |
| 0 | NaN | 5.0 |
| 1 | Güzel. Ayağım 34 numara, 36 rahat oldu. Almayı... | 5.0 |
| 2 | Ürün birebir aynı geldi, yorumlarda okuduğum g... | 5.0 |
| 3 | Urun cok guzel, ayagi genis olanlar icin bir b... | 5.0 |
| 4 | Uzun süre giyip yurumedim henüz. Evde denediği... | 5.0 |

#Veri önişleme (Data Preprocessing) :
- Veri önişleme adımları
    1. Eksik (NaN) değerlerin değerlendirilmesi 
    2. Noktalama İşaretlerinin kaldırılması
    3. Sayısal değerlerin kaldırılması
    4. Tüm karakterlerin küçük harfe dönüştürme

#Veri önişleme adımlarını tamamladıktan sonra elde ettiğimiz çıktı aşağıdadır.

| index | Yorum | Yıldız |
| --- | --- | --- |
| 1 | güzel ayağım numara rahat oldu almayı düşünenler bunu göz önünde bulundursun ben memnun kaldım öneririm | 5.0 |
| 2 | ürün birebir aynı geldi yorumlarda okuduğum gibi dar kalıptı bunu düşünerek alın kargo hızlıydı elimize çabuk ulaştı | 5.0 |
| 3 | urun cok guzel ayagi genis olanlar icin bir beden buyuk olabilir | 5.0 |
| 4 | uzun süre giyip yurumedim henüz evde denediğim kadarıyla rahat yorumlara bakarak numara büyük almıştım kendi numaramı alsam olurmuş ama ragatsiz edecek bir durum gibi durmuyor ayakta duruşu çok sik | 5.0 |
| 5 | çok kaliteli bir ürün anneme aldıktan sonra bir çift de kardeşime aldım hem fp ürünü hemde şık | 5.0 |


#Veri Görselleştirme

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

![newplot.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8270f11f-8742-4fa4-9cf2-d19eae4dbfa1/newplot.png)

![newplot (6).png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0b46b3f2-1230-4065-b6ce-4682117ab72f/newplot_(6).png)

Word2Vec Nedir?

Tahmin tabanlı (prediction-based) kelime temsil yöntemi olup, 2013 yılında Google araştırmacısı Thomas Mikolov ve ekip arkadaşları ile birlikte temelinde yapay sinir ağı ile iki farklı model kullanarak kelimelerin eğitilmesi amaçlanıp geliştirilmiştir.

Word2Vec’in kullandığı iki model CBOW(Continuous Bag of Words) ve Skip-Gram Model’dir. Bu iki modelin mimarisini inceleyecek olursak:

Continuous Bag of Words: CBOW modelinde pencere boyutu merkezinde olmayan kelimeler girdi olarak alınıp, merkezinde olan kelimeler çıktı olarak tahmin edilmeye çalışılmaktadır. Bu durum aşağıdaki şekilde gösterilmeye çalışılmıştır. Burada w(t) ile gösterilen değer, cümlenin merkezinde bulunan ve tahmin edilmek istenen çıktı değeri iken, w(t-2)…..w(t+2) ile gösterilen değerler ise tercih edilen pencere boyutuna göre merkezde olmayan çıktı değerleridir.

![image](https://user-images.githubusercontent.com/100937634/200265589-b001a920-4d86-4395-b6b2-711fe6f0f8f8.png)

Skip Gram: Skip Gram modelinde pencere boyutu merkezinde olan kelimeler girdi olarak alınıp, merkezinde olmayan kelimeler çıktı olarak tahmin edilmeye çalışılmaktadır.Bu durum aşağıdaki şekilde gösterilmeye çalışılmıştır. Burada w(t) ile gösterilen değer, cümlenin merkezinde bulunan ve girdi değeri iken, w(t-2)…..w(t+2) ile gösterilen değerler ise cümlenin merkezinde olmayan tercih edilen pencere boyutuna göre tahmin edilmek istenen çıktı değerleridir.
Skip Gram model ve CBOW arasındaki tek fark Skip Gram modelin CBOW’un tam tersi olmasıdır. Yani, yapay sinir ağında çıktılar ve girdilerin yeri değiştirmektedir.

![image](https://user-images.githubusercontent.com/100937634/200266361-7b08489b-9381-4585-a331-0e0aa6c4f779.png)


![newplot (2) (1)](https://user-images.githubusercontent.com/100937634/200258042-5b11a879-066f-47d8-9f26-156cbbfffbe0.png)

![newplot (4).png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/197b54cb-aefa-4dd9-a0f8-ae33e52be557/newplot_(4).png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3e898255-97f9-4836-acbc-4958ca598b27/Untitled.png)

#CBOW Model Sonuçları

Cosine similarity between 'kutuda' ve  'muhafaza' - CBOW :  0.091992885

Cosine similarity between 'kutuda' ve  'etmenize' - CBOW :  -0.19237778

#Skip-Gram Model Sonuçları

Cosine similarity between 'kutuda' ve  'muhafaza' - Skip Gram :  0.08502861

Cosine similarity between 'kutuda' ve 'etmenize' - Skip Gram :  -0.17239794

Sonuçlar incelenirse;"kutuda" kelimesinden sonra "muhafaza" kelimesinin kaç olasılıkla geleceğini tahmin eder.
CBOW modelleri genel olarak küçük datasetlerde daha iyi çalışırken, büyük datasetlerde Skip-gram daha iyi çalışmaktadır. CBOW daha az computation power gerektirirken, Skip-Gram daha fazla computation power gerektirir. CBOW 2 veya daha çok anlamlı kelimeleri anlamakta iyi değilken Skip-Gram 2 veya daha çok anlamlı kelimeleri daha iyi öğrenebilmektedir.
