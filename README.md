# NLP-WORD2VEC
 Doğal Dİl İşleme
**Python ile [Hepsiburada.com](http://Hepsiburada.com) ‘dan Verilerin Kazınması**

Projeden bahsetmek gerekirse hepsiburada da olan kadın ayakkabılarına yapılan yorumlar ve o yorumlara gelen yıldızları çekme işlemi ve Nlp modellerinden 
word2vec modelini kullanarak projeyi geliştirdim.

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


#Veri Görselleştirme
![newplot (2) (1)](https://user-images.githubusercontent.com/100937634/200258042-5b11a879-066f-47d8-9f26-156cbbfffbe0.png)

