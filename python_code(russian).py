# Подгружаем необходимые библиотеки
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances
import umap
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

# Загрузка данных
edits = pd.read_csv(".../Editorials-about-Russia-2024.csv")
edits = edits.dropna()

# Простой анализ заголовков (вместо udpipe для японского)
titles = edits['title'].astype(str)
keywords = ["ロシア", "露", "モスクワ", "プーチン"]
title_rus = titles.str.extract(f"({'|'.join(keywords)})", expand=False).dropna()

title_rus_counts = title_rus.value_counts()
title_rus_counts.plot(kind='bar')
plt.show()

# Подготовка текста (упрощенно, без udpipe и стоп-слов для японского)
# Замените этот блок на более продвинутую обработку японского текста, если необходимо
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', str(text)) # Убираем пунктуацию
    text = text.lower() # Приводим к нижнему регистру
    return text

edits['processed_text'] = edits['text'].apply(preprocess_text)

# Анализ частотности слов (вместо wordcloud)
all_words = ' '.join(edits['processed_text']).split()
word_counts = Counter(all_words)
most_common_words = word_counts.most_common(100)

wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(most_common_words))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# LSA (латентно-семантический анализ)
vectorizer = TfidfVectorizer()
dtm = vectorizer.fit_transform(edits['processed_text'])
lsa = TruncatedSVD(n_components=50, random_state=0)
lsa_space = lsa.fit_transform(dtm)

# Поиск ближайших слов (пример)
word_embeddings = lsa.components_.T
word_index = vectorizer.vocabulary_
reverse_word_index = {v: k for k, v in word_index.items()}

def nearest_word(word, embeddings, index, reverse_index, topn=20):
    word_vector = embeddings[index[word]]
    distances = cosine_distances(word_vector.reshape(1, -1), embeddings)[0]
    closest_indices = np.argsort(distances)[1:topn+1]  # Exclude the word itself
    return [reverse_index[i] for i in closest_indices]

print(nearest_word("ロシア", word_embeddings, word_index, reverse_word_index))
print(nearest_word("モスクワ", word_embeddings, word_index, reverse_word_index))

# UMAP визуализация
umap_data = umap.UMAP(n_neighbors=3, min_dist=0.3, random_state=0).fit_transform(lsa_space)
plt.scatter(umap_data[:, 0], umap_data[:, 1])
plt.show()

# LDA (латентное размещение Дирихле)
lda_model = LatentDirichletAllocation(n_components=20, random_state=0)
lda_model.fit(dtm)

# Вывод топ-слов для каждой темы
for topic_idx, topic in enumerate(lda_model.components_):
    print(f"Topic #{topic_idx}:")
    top_words_idx = topic.argsort()[:-12 - 1:-1]
    top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
    print(" ".join(top_words))
