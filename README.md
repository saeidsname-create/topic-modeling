# Project Title  
This project was created to be able to produce top modeling of paragraphs.# Text Chunking, Topic Modeling & Clustering
کد مورد توضیح:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from collections import Counter


text = """
پاراگراف اول درباره آب و هوا و بارش باران است.
پاراگراف دوم درباره سیاست و دولت و انتخابات است.
پاراگراف سوم درباره فوتبال و ورزش و لیگ برتر است.
پاراگراف چهارم درباره تیم های ورزشی و فوتبال اروپاست.
پاراگراف پنجم درباره دولت و مجلس و سیاست داخلی است.
"""

chunks = [p.strip() for p in text.split("
") if p.strip()]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(chunks)

k = 3   
model = KMeans(n_clusters=k, random_state=0)
labels = model.fit_predict(X)


cluster_groups = {i: [] for i in range(k)}
for i, lbl in enumerate(labels):
    cluster_groups[lbl].append(chunks[i])


def extract_topic(paragraphs, top_n=3):
    words = " ".join(paragraphs).split()
    common = Counter(words).most_common(top_n)
    return [w for w, count in common]

cluster_topics = {}
for c in cluster_groups:
    cluster_topics[c] = extract_topic(cluster_groups[c], top_n=3)


for c in cluster_groups:
    print(f"خوشه {c}:")
    print("  پاراگراف‌ها:")
    for p in cluster_groups[c]:
        print("    -", p)
    print("  تاپیک:", ", ".join(cluster_topics[c]))
    print()
```

---


# توضیح خط‌به‌خط کد

## وارد کردن کتابخانه‌ها

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from collections import Counter
```

* `CountVectorizer`: تبدیل پاراگراف‌ها به بردارهای Bag-of-Words.
* `KMeans`: الگوریتم خوشه‌بندی.
* `Counter`: استخراج پرتکرارترین کلمات هر خوشه.

---

## تعریف متن نمونه و ساخت چانک‌ها

```python
text = """
...
"""
chunks = [p.strip() for p in text.split("
") if p.strip()]
```

**توضیح:**

1. متن چند خطی تعریف شده.
2. `split("
   ")` هر خط را جدا می‌کند.
3. با `strip()` فاصله‌های اضافی حذف می‌شود.
4. خطوط خالی فیلتر می‌شوند.

در نهایت `chunks` یک لیست مثل این می‌سازد:

```
[
 "پاراگراف اول ...",
 "پاراگراف دوم ...",
 ...
]
```
**جایگزین:**
* از کتابخانه های regex و hazm, nltk هم میشد استفاده کرد ولی هر کدوم از آنها مشکلاتی دارند:
 1. regex : مشکلش این است که برای اینکه ما بتوانیم کلمات را حذف کنیم باید دقیق کلمات را بگوییم تا حذف شود و برای هر کدام یک دستوری بنویسیم تا مثلا حرف بهترین و بدترین حذف شود
 2. hazm : این کتابخونه هم بخاطر اینکه دانلودش سخته و نرم افزار های pycharm , vscode سخت دانلود میشه بخاطر همین نمیشه استفاده کرد و همین طور پایتون 3.13 به بالا را پشتیبانی نمیکند
 3. nltk : این کتابخونه هم چون فقط کلمات انگلیسی را پشتیبانی میکند نمیتوان استفاده کرد

---

## بردارسازی متن‌ها

```python
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(chunks)
```

* **CountVectorizer** هر پاراگراف را به شمارش کلمات تبدیل می‌کند.
* خروجی `X` یک ماتریس sparse است با شکل:
  **(تعداد پاراگراف‌ها) × (تعداد کلمات یکتا)**.

**چرا این روش؟**
چون برای KMeans لازم است متن‌ها را تبدیل به بردارهای عددی کنیم. CountVectorizer ساده، سریع و مناسب متن‌های کوتاه است.

**جایگزین‌ها:**


* Word Embeddings (مثل FastText یا Word2Vec)
* این بردار ها برای kmeans خیلی خوب تر از countvectorizer هستن و حدود 100 بعد را تشکیل میدهند که از تابع ما بیشتر است
* Sentence Embeddings (BERT, etc.)
* این تابع از کتابخونه ی sentence_transformers از کتابخونه ی gensim هم بهتر است و حدود 348 بعد را تشکیل میدهد که برای متن های بزرگ خیلی خوب است 

---

## خوشه‌بندی با KMeans

```python
k = 3
model = KMeans(n_clusters=k, random_state=0)
labels = model.fit_predict(X)
```

* `k = 3`: تعداد خوشه‌ها را مشخص می‌کند.
* `KMeans(...)`: مدل خوشه‌بندی ساخته می‌شود.
* `fit_predict(X)`:
  مدل روی داده‌ها آموزش می‌بیند و برای هر پاراگراف یک برچسب خوشه تولید می‌کند.

### گزینه‌های جایگزین

* DBSCAN (تشخیص نویز و شکل‌های غیرکروی)
* Agglomerative Clustering (خوشه‌بندی سلسله‌مراتبی)
* HDBSCAN (برای داده‌های واقعی بهتر است)
* این ها روش های خوبی هستن ولی چون متن ما کوچک هستش و خیلی متن بلندی نداریم همین kmeans بهتر هستش

---

## گروه‌بندی پاراگراف‌ها بر اساس خوشه

```python
cluster_groups = {i: [] for i in range(k)}
for i, lbl in enumerate(labels):
    cluster_groups[lbl].append(chunks[i])
```

* یک دیکشنری ساخته می‌شود که برای هر خوشه یک لیست دارد.
* حلقهٔ `for`: هر پاراگراف را در خوشهٔ خودش قرار می‌دهد.

خروجی نمونه:

```
{
 0: [پاراگراف‌های مربوط به موضوع 1],
 1: [پاراگراف‌های مربوط به موضوع 2],
 ...
}
```

---

## استخراج تاپیک ساده از هر خوشه

```python
def extract_topic(paragraphs, top_n=3):
    words = " ".join(paragraphs).split()
    common = Counter(words).most_common(top_n)
    return [w for w, count in common]
```

### توضیح:

1. تمام پاراگراف‌های خوشه در یک رشته ادغام می‌شوند.
2. همهٔ کلمات جدا می‌شوند.
3. با `Counter` پرتکرارترین کلمات انتخاب می‌شوند.
4. فقط خود کلمات برگشت داده می‌شوند.

### نکته مهم

این **تاپیک‌مدلینگ واقعی نیست**.
فقط «سه کلمهٔ پرتکرار» است که نقش «خلاصه موضوعی» را بازی می‌کند.

### جایگزین‌های حرفه‌ای‌تر

* LDA (Latent Dirichlet Allocation)
* NMF Topic Modeling
* BERTopic (مبتنی بر embeddings)

---

## تولید تاپیک هر خوشه

```python
cluster_topics = {}
for c in cluster_groups:
    cluster_topics[c] = extract_topic(cluster_groups[c], top_n=3)
```

* برای هر خوشه سه کلمهٔ پرتکرار انتخاب می‌شود.

---

## چاپ خروجی نهایی

```python
for c in cluster_groups:
    print(f"خوشه {c}:")
    print("  پاراگراف‌ها:")
    for p in cluster_groups[c]:
        print("    -", p)
    print("  تاپیک:", ", ".join(cluster_topics[c]))
    print()
```

این بخش خروجی نهایی را مانند زیر چاپ می‌کند:

```
خوشه 0:
  پاراگراف‌ها:
    - پاراگراف اول ...
    - پاراگراف سوم ...
  تاپیک: فوتبال, ورزش, ...

خوشه 1:
  پاراگراف‌ها:
    - پاراگراف دوم ...
    - پاراگراف پنجم ...
  تاپیک: دولت, سیاست, ...
```

---

# 3. جمع‌بندی

این کد یک نسخهٔ ساده از خوشه‌بندی متن است که برای پروژه‌های مقدماتی NLP کاملاً قابل قبول است.
مراحل اصلی:

1. تقسیم متن
2. بردارسازی با CountVectorizer
3. خوشه‌بندی با KMeans
4. استخراج تاپیک با شمارش پرتکرارترین کلمات

اگر نیاز داری نسخهٔ حرفه‌ای‌تر با LDA یا مدل‌های BERT اضافه کنم، فقط بگو.
