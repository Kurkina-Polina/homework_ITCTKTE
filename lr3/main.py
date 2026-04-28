from textrank import TextRankSummarizer
from dataset_loader import load_gazeta_dataset
from evaluate import calculate_rouge, print_evaluation

def main():
    # Переключатель режима:
    # True - запустить тест на датасете с оценкой ROUGE
    # False - запустить режим сдачи
    TEST_MODE = False

    summarizer = TextRankSummarizer(limit=300)

    if TEST_MODE:
        print("Запуск тестирования на датасете Gazeta...")
        texts, references = load_gazeta_dataset(n_samples=20)

        if not texts:
            print("Не удалось загрузить датасет. Проверьте интернет.")
            return

        candidates = []
        for text in texts:
            summary = summarizer.summarize(text)
            candidates.append(summary)

        # Считаем метрики
        scores = calculate_rouge(candidates, references)
        print_evaluation(scores)

    else:
        # Интерактивный режим
        input_text = input("Введите текст для суммаризации:\n\n")

        summarizer = TextRankSummarizer(limit=300)
        summary = summarizer.summarize(input_text)

        print("\n--- Реферат ---")
        print(summary)
        print(f"\nДлина: {len(summary)} символов")

if __name__ == "__main__":
    main()