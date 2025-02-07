#Загружаем и привязываем необходимые библиотеки
library(tidyverse)
library(tidytext)
library(udpipe)
library(stopwords)
library(RColorBrewer)
library(wordcloud)
library(philentropy)
library(irlba)
library(uwot)
library(topicmodels)
library(furrr)
library(ldatuning)

#Открываем файл
edits <- read_csv(".../Editorials-about-Russia-2024.csv")

edits <- edits |> 
  drop_na()

#Аннотируем текст
udpipe_download_model(language = "japanese-gsd")
japanese_gsd <- udpipe_load_model(file = "japanese-gsd-ud-2.5-191206.udpipe")
edit_annotate <- udpipe_annotate(japanese_gsd, edits$text, trace = TRUE)

edit_pos <- as.tibble(edit_annotate)

#Фильтруем текст, избавляемся от "стоп-слов"
edit_pos <- edit_pos|> 
  select(-paragraph_id) |> 
  filter(upos != 'PUNCT')

stop_words <- c(stopwords::stopwords("ja", source = "marimo"), 
                stopwords::stopwords("ja", source = "stopwords-iso"))

other <- c('氏', '年', '目', '以上', '間', 'ため')

edit_tidy <- edit_pos |>
  filter(!token %in% stop_words) |> 
  filter(!token %in% other)

#Анализируем заголовки
titles <- edits |>
  select(title) |> 
  sapply(as.character)

sum(str_detect(titles, "ロシア|露|モスクワ|プーチン"))

title_rus <- titles |> 
  str_extract("ロシア|露|モスクワ|プーチン")|>
  as.tibble() |>  
  drop_na()

title_rus |> 
  ggplot(aes(value, fill = value)) +
  geom_bar(show.legend = FALSE) +
  coord_flip() +
  xlab(NULL) +
  ylab(NULL) 

#Рассматриваем абсолютную частотность существительных
edit_nouns <- edit_tidy |> 
  filter(upos %in% c("NOUN")) |> 
  select(doc_id, lemma)

pal <- RColorBrewer::brewer.pal(8, "Dark2")

nouns <- edit_nouns |> 
  count(lemma) |>
  arrange(-n)

nouns_wc <- wordcloud(nouns$lemma, nouns$n, colors = pal, max.words = 100, 
                      rot.per = 0)

edit_propn <- edit_tidy |> 
  filter(upos %in% c("PROPN")) |> 
  select(doc_id, lemma)

propn <- edit_propn |>
  count(lemma) |> 
  arrange(-n)

propn_wc <- wordcloud(propn$lemma, propn$n, colors = pal, max.words = 100, 
                      rot.per = 0)

#Подготавливаем данные к LSA
edit_imp <- edit_tidy|> 
  filter(upos %in% c("PROPN", "NOUN")) |> 
  select(doc_id, lemma)

rare_words <- edit_imp |> 
  distinct(doc_id, lemma) |>
  count(lemma) |> 
  filter(n == 1) |> 
  pull(lemma)

edit_pruned <- edit_imp |> 
  filter(!lemma %in% rare_words)

edit_counts <- edit_pruned |> 
  count(lemma, doc_id) |> 
  bind_tf_idf(lemma, doc_id, n) |> 
  select(-n, -tf, -idf)

#Создаем матрицу "термин-документ"
dtm <- edit_counts |> 
  cast_sparse(lemma, doc_id, tf_idf)

k <- 50
set.seed(08132025)
lsa_space<- irlba(dtm, nv = k, maxit = 500)

rownames(lsa_space$u) <- rownames(dtm)
colnames(lsa_space$u) <- paste0("dim", 1:k)

word_emb <- lsa_space$u |> 
  as.data.frame() 

dist_word_mx <- word_emb  |> 
  philentropy::distance(method = "cosine", use.row.names = TRUE) 

#Создаем функцию для поиска ближайших слов
nearest_word <- function(dist_word_mx, word, number = 20) {
  sort(dist_word_mx[word, ], decreasing = TRUE) |> 
    head(number) |> 
    names()
}

nearest_word(dist_word_mx, "露")
nearest_word(dist_word_mx, "モスクワ")

# Визуализируем результат кластеризации статей
set.seed(07062024)
viz_lsa <- umap(lsa_space$v ,  n_neighbors = 3) 
tibble(doc = rownames(viz_lsa),
       V1 = viz_lsa[, 1], 
       V2 = viz_lsa[, 2]) |> 
  ggplot(aes(x = V1, y = V2, label = doc)) + 
  geom_text(size = 3, alpha = 0.8, position = position_jitter(width = 1.5, height = 1.5)) +
  theme_light()

#Подготавливаем данные для LDA 
edit_count <- edit_pruned |> 
  count(lemma, doc_id)

edit_dtm <- edit_count |> 
  cast_dtm(doc_id, term = lemma, value = n)

#Проверяем число тем
plan(multisession, workers = 6)

n_topics <- c(2, 5, 10, 15, 20, 25)
edit_lda_models <- n_topics  |> 
  future_map(topicmodels::LDA, x = edit_dtm, 
             control = list(seed = 0211), .progress = TRUE)


data_frame(k = n_topics,
           perplex = map_dbl(edit_lda_models, perplexity))  |> 
  ggplot(aes(k, perplex)) +
  geom_point() +
  geom_line() +
  labs(title = "Оценка LDA модели",
       x = "Число топиков",
       y = "Perplexity")

result <- FindTopicsNumber(
  edit_dtm,
  topics = n_topics,
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 05092024),
  verbose = TRUE
)

FindTopicsNumber_plot(result)

#Создаём модель
edit_lda <- topicmodels::LDA(edit_dtm, k = 20, control = list(seed = 05092024))

edit_topics <- tidy(edit_lda, matrix = "beta")

#Визуализируем главные слова-компоненты
edit_top_terms_1_10 <- edit_topics |> 
  filter(topic < 11) |> 
  group_by(topic) |> 
  arrange(-beta) |> 
  slice_head(n = 12) |> 
  ungroup()

edit_top_terms_1_10 |> 
  mutate(term = reorder(term, beta)) |> 
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) + 
  facet_wrap(~ topic, scales = "free", ncol=4) +
  coord_flip()

edit_top_terms_11_20 <- edit_topics |> 
  filter(topic > 10) |> 
  group_by(topic) |> 
  arrange(-beta) |> 
  slice_head(n = 12) |> 
  ungroup()

edit_top_terms_11_20 |> 
  mutate(term = reorder(term, beta)) |> 
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) + 
  facet_wrap(~ topic, scales = "free", ncol=4) +
  coord_flip()
