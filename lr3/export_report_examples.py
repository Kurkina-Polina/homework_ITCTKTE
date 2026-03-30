import pandas as pd
from textrank import TextRankSummarizer
from dataset_loader import load_gazeta_dataset
from evaluate import calculate_rouge

def export_examples(n_samples=50):
    texts, references = load_gazeta_dataset(n_samples=n_samples)

    if not texts:
        print("Ошибка загрузки датасета!")
        return

    summarizer = TextRankSummarizer(limit=300)
    results = []

    for i, (text, gold) in enumerate(zip(texts, references)):
        predicted = summarizer.summarize(text)
        scores = calculate_rouge([predicted], [gold])

        results.append({
            "№": i + 1,
            "Исходный текст": text[:200] + "...",
            "Эталонный реферат": gold,
            "Сгенерированный реферат": predicted,
            "R1-F1": round(scores['rouge1'], 4),
            "R2-F1": round(scores['rouge2'], 4),
            "RL-F1": round(scores['rougeL'], 4),
            "Длина": len(predicted)
        })

    df = pd.DataFrame(results)
    df['AVG-F1'] = round((df['R1-F1'] + df['R2-F1'] + df['RL-F1']) / 3, 4)
    df_sorted = df.sort_values(by='AVG-F1', ascending=False)

    cols = ['Эталонный реферат', 'Сгенерированный реферат', 'R1-F1', 'R2-F1', 'RL-F1']

    df_sorted.to_html('report_examples.html', encoding='utf-8', index=False)


if __name__ == "__main__":
    export_examples(n_samples=50)