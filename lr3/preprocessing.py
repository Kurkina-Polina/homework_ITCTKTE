import re
import string
import nltk
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
    return [w for w in words if w not in stop_words and len(w) > 2]