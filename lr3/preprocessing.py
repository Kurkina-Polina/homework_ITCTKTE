import re
import string
import nltk
import math
from nltk.corpus import stopwords
import pymorphy3
morph = pymorphy3.MorphAnalyzer()

# Попытка загрузить стоп-слова, если не получилось — скачиваем
try:
    STOP_WORDS = set(stopwords.words('russian'))
except LookupError:
    # Если словарь не найден, скачиваем его автоматически
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('russian'))

def split_sentences(text):
    """Разбивает текст на предложения"""
    # Разделяем по точкам, восклицательным и вопросительным знакам
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def tokenize(text, custom_stopwords=None):
    """Превращает предложение в список чистых слов"""
    # Используем стоп-слова из NLTK, но можно добавить свои вручную
    stop_words = STOP_WORDS
    if custom_stopwords:
        stop_words = stop_words.union(set(custom_stopwords))

    text = text.lower()
    # Удаляем знаки препинания
    text = text.translate(str.maketrans('', '', string.punctuation + '«»""„"—'))
    words = text.split()

    processed_words = []
    for w in words:
        if w not in stop_words and len(w) > 2:
            # Лемматизация: приводим к начальной форме
            processed_words.append(morph.parse(w)[0].normal_form)

    # Фильтруем стоп-слова и слишком короткие слова
    return processed_words

def get_word_idf(sentences_tokens):
    """
    Вычисляет IDF для каждого слова среди всех предложений.
    IDF = log(N / df), где N - всего предложений, df - в скольких предложениях встречается слово
    """
    n_docs = len(sentences_tokens)
    if n_docs == 0:
        return {}

    # В скольких предложениях встречается каждое слово
    word_doc_freq = {}
    for tokens in sentences_tokens:
        unique_words = set(tokens)
        for word in unique_words:
            word_doc_freq[word] = word_doc_freq.get(word, 0) + 1

    # Вычисляем IDF с лапласовским сглаживанием (+1 для предотвращения деления на ноль)
    idf = {}
    for word, df in word_doc_freq.items():
        idf[word] = math.log((n_docs + 1) / (df + 1)) + 1

    return idf

def calculate_sentence_tfidf_scores(sentences_tokens, idf_weights):
    """
    Рассчитывает TF-IDF score для каждого предложения.
    Формула: Σ (count(w)/|s|) * IDF(w) * (|s|**0.5)
    """
    scores = []
    for tokens in sentences_tokens:
        if len(tokens) == 0:
            scores.append(0.0)
            continue

        # TF: частота слова в предложении
        word_freq = {}
        for word in tokens:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Сумма TF-IDF для всех слов
        tfidf_sum = 0.0
        for word, count in word_freq.items():
            tf = count / len(tokens)
            idf = idf_weights.get(word, 1.0)
            tfidf_sum += tf * idf * (len(tokens)**0.5)

        scores.append(tfidf_sum)

    return scores