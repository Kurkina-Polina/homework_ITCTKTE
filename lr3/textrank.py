import math
from preprocessing import split_sentences, tokenize

class TextRankSummarizer:
    def __init__(self, limit=300):
        self.limit = limit
        self.damping = 0.85
        self.iterations = 20

    def _sentence_similarity(self, sent1, sent2):
        """Вычисляет сходство двух предложений"""
        if not sent1 or not sent2:
            return 0.0

        intersection = set(sent1).intersection(set(sent2))

        if len(intersection) == 0:
            return 0.0

        # Формула сходства (нормализация по длине)
        return len(intersection) / (len(sent1) + len(sent2))

    def _build_graph(self, sentences_tokens):
        """Создает матрицу сходства"""
        n = len(sentences_tokens)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                matrix[i][j] = self._sentence_similarity(sentences_tokens[i], sentences_tokens[j])
        return matrix

    def _pagerank(self, matrix):
        """Алгоритм PageRank"""
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

    def summarize(self, text):
        """Основной метод суммаризации"""
        sentences = split_sentences(text)
        if not sentences:
            return ""

        # Токенизация (стоп-слова уже внутри функции tokenize)
        sentences_tokens = [tokenize(s) for s in sentences]

        matrix = self._build_graph(sentences_tokens)
        scores = self._pagerank(matrix)

        # Сортировка предложений по важности
        ranked_sentences = sorted(
            [(i, sentences[i], score) for i, score in enumerate(scores)],
            key=lambda x: x[2],
            reverse=True
        )

        # Сборка реферата с ограничением по символам
        summary = []
        current_length = 0

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