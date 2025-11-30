# Project Title  
This project was created to be able to produce top modeling of paragraphs.# Text Chunking, Topic Modeling & Clustering

این پروژه یک متن ورودی بدون محدودیت کاراکتر را دریافت می‌کند، متن را بر اساس پاراگراف‌ها چانک‌بندی می‌کند، روی هر چانک تاپیک‌مدلینگ انجام می‌دهد و در نهایت چانک‌ها را خوشه‌بندی می‌کند. همچنین برای هر خوشه، پرتکرارترین تاپیک استخراج می‌شود.

## Features
- دریافت متن ورودی بدون محدودیت حجم
- چانک‌بندی متن براساس پاراگراف
- تولید بردارهای عددی با CountVectorizer
- اجرای Topic Modeling با LDA
- خوشه‌بندی چانک‌ها با KMeans
- استخراج تاپیک غالب هر خوشه
## Example
text = """متن طولانی شما..."""

chunks = chunk_text(text)
topics = extract_topics(chunks)
clusters = cluster_chunks(chunks, topics)

print(clusters)


chunk_text
متن را بر اساس پاراگراف تقسیم می‌کند و یک لیست از چانک‌ها می‌سازد.

extract_topics
برای هر چانک یک بردار احتمال تاپیک‌ها تولید می‌کند (نه یک تاپیک واحد) و آن‌ها را در یک لیست قرار می‌دهد.

cluster_chunks
با استفاده از همان بردارهای تاپیک، چانک‌ها را با مدل KMeans خوشه‌بندی می‌کند و برچسب خوشه هر چانک را تولید می‌کند.

print(clusters)
نتیجه نهایی خوشه‌بندی چاپ می‌شود.


## Requirements
برای اجرای این پروژه، کتابخانه‌ زیر لازم است:

```bash
pip install scikit-learn
 
