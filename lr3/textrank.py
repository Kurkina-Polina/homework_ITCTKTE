import math
import re
from preprocessing import split_sentences, tokenize, get_word_idf, calculate_sentence_tfidf_scores

class TextRankSummarizer:
    def __init__(self, limit=300):
        self.limit = limit
        self.damping = 0.85
        self.iterations = 20
        self.tfidf_weight = 0.5
        self.pagerank_weight = 0.5

    def _sentence_similarity(self, sent1, sent2, idf_weights):
        """ Вычисляет сходство двух предложений. """
        if not sent1 or not sent2:
            return 0.0

        # Находим общие слова
        intersection = set(sent1).intersection(set(sent2))

        if len(intersection) == 0:
            return 0.0

        # Сумма весов общих слов (чем реже слово, тем выше вес)
        weighted_intersection = sum(idf_weights.get(word, 1.0) for word in intersection)

        # Нормализуем на общую длину предложений
        # Можно также взвешивать длину, но базовая версия работает хорошо
        normalization = len(sent1) + len(sent2)

        return weighted_intersection / normalization

    def _build_graph(self, sentences_tokens, idf_weights):
        """Создает матрицу сходства между предложениями"""
        n = len(sentences_tokens)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                matrix[i][j] = self._sentence_similarity(
                    sentences_tokens[i],
                    sentences_tokens[j],
                    idf_weights
                )
        return matrix

    def _pagerank(self, matrix):
        """Алгоритм PageRank для ранжирования узлов графа"""
        n = len(matrix)
        if n == 0: return []

        ranks = [1.0 / n] * n

        for _ in range(self.iterations):
            new_ranks = [0.0] * n
            for i in range(n):
                neighbor_sum = sum(matrix[j][i] * ranks[j] for j in range(n))
                new_ranks[i] = (1 - self.damping) + self.damping * neighbor_sum
            ranks = new_ranks

        return ranks

    def _normalize_scores(self, scores):
        """Нормализует_scores в диапазон [0, 1] для комбинации"""
        if not scores or max(scores) == 0:
            return scores
        max_score = max(scores)
        return [s / max_score for s in scores]

    def summarize(self, text):
        """Основной метод суммаризации"""
        sentences = split_sentences(text)
        sentences = [s for s in sentences if 25 <= len(s) <= 300]
        dateline_pattern = re.compile(r'^[А-Яа-яЁё]+\s*,\s*\d+\s+[а-яА-ЯЁё]+')
        sentences = [s for s in sentences if not dateline_pattern.match(s)]
        if not sentences:
            return ""

        # Токенизируем
        sentences_tokens = [tokenize(s) for s in sentences]

        # Вычисляем IDF веса для всего текста
        idf_weights = get_word_idf(sentences_tokens)

        #  Считаем TF-IDF для каждого предложения
        tfidf_scores = calculate_sentence_tfidf_scores(sentences_tokens, idf_weights)


        # Строим граф с учетом TF-IDF
        matrix = self._build_graph(sentences_tokens, idf_weights)
        pagerank_scores = self._pagerank(matrix)

        tfidf_norm = self._normalize_scores(tfidf_scores)
        pagerank_norm = self._normalize_scores(pagerank_scores)

        # Комбинируем TF-IDF + PageRank
        final_scores = [
            self.tfidf_weight * t + self.pagerank_weight * p
            for t, p in zip(tfidf_norm, pagerank_norm)
        ]

        # Позиционный бонус
        for i in range(len(final_scores)):
            if i == 0:
                final_scores[i] *= 2
            elif i == 1:
                final_scores[i] *= 1.5
            elif i == 2:
                final_scores[i] *= 1.2
            elif i >= len(final_scores) - 2:
                final_scores[i] *= 1.1
            elif i == len(final_scores) - 1:
                final_scores[i] *= 1.5
        # Сначала отбираем топ-N по важности
        top_n = sorted(
            [(i, sentences[i], score) for i, score in enumerate(final_scores)],
            key=lambda x: x[2],
            reverse=True
        )

        # Затем сортируем отобранные по исходному порядку (по индексу i)
        ranked_sentences = sorted(top_n, key=lambda x: x[0])

        # Сборка реферата с ограничением по символам
        summary = []
        current_length = 0

        # Принудительное включение первого предложения повышает оценки
        if sentences:
            first_sent = sentences[0]
            if len(first_sent) <= self.limit:
                summary.append(first_sent)
                current_length = len(first_sent)

        for idx, original_text, score in ranked_sentences:
            # +1 на пробел между предложениями
            add_len = len(original_text) + (1 if summary else 0)

            if current_length + add_len <= self.limit:
                summary.append(original_text)
                current_length += add_len
            else:
                # Если первое предложение не влезает, обрезаем его
                if not summary:
                    remainder = self.limit - current_length
                    if remainder > 10:
                        # Обрезаем по слову, чтобы не было обрубков
                        trimmed = original_text[:remainder].rsplit(' ', 1)[0]
                        summary.append(trimmed + '...')
                break

        return " ".join(summary)