import re
import pymorphy3
from rouge_score import rouge_scorer

morph = pymorphy3.MorphAnalyzer()

class Tokenizer:
    @staticmethod
    def tokenize(text: str) -> list:
        """Корректная токенизация русского текста с лемматизацией"""
        words = re.findall(r'\w+', text.lower())
        return [morph.parse(w)[0].normal_form for w in words if len(w) > 2]

def calculate_rouge(candidates, references):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False,
        tokenizer=Tokenizer
    )

    avg_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

    for cand, ref in zip(candidates, references):
        scores = scorer.score(ref, cand)
        for key in avg_scores:
            avg_scores[key] += scores[key].fmeasure

    n = len(candidates)
    if n > 0:
        for key in avg_scores:
            avg_scores[key] /= n

    return avg_scores

def print_evaluation(scores):
    print("\n--- Оценка ROUGE ---")
    print(f"ROUGE-1: {scores['rouge1']:.4f}")
    print(f"ROUGE-2: {scores['rouge2']:.4f}")
    print(f"ROUGE-L: {scores['rougeL']:.4f}")
    print("--------------------")