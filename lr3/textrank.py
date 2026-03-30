import math
from preprocessing import split_sentences, tokenize, get_word_idf

class TextRankSummarizer:
    def __init__(self, limit=300):
        self.limit = limit
        self.damping = 0.85
        self.iterations = 20

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

    def summarize(self, text):
        """Основной метод суммаризации"""
        sentences = split_sentences(text)
        if not sentences:
            return ""

        # Токенизируем
        sentences_tokens = [tokenize(s) for s in sentences]

        # Вычисляем IDF веса для всего документа ---
        idf_weights = get_word_idf(sentences_tokens)

        # Строим граф с учетом TF-IDF
        matrix = self._build_graph(sentences_tokens, idf_weights)
        scores = self._pagerank(matrix)

        # Позиционный бонус
        for i in range(len(scores)):
            if i == 0:
                scores[i] *= 1.5  # Первое предложение +50%
            elif i == len(scores) - 1:
                scores[i] *= 1.2  # Последнее +20%
            elif i < 3:
                scores[i] *= 1.1  # Первые три +10%

        # Сортировка предложений по важности
        ranked_sentences = sorted(
            [(i, sentences[i], score) for i, score in enumerate(scores)],
            key=lambda x: x[2] / math.sqrt(len(x[1]) + 1), # жадный отбор
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