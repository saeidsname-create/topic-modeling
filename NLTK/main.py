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

chunks = [p.strip() for p in text.split("\n") if p.strip()]

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
