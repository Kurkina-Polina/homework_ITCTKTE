from rouge_score import rouge_scorer

def calculate_rouge(candidates, references):
    """
    candidates: список рефератов от алгоритма
    references: список эталонных рефератов (из датасета)
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    avg_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

    for cand, ref in zip(candidates, references):
        scores = scorer.score(ref, cand) # Важно: сначала эталон, потом кандидат
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