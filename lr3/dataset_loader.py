from datasets import load_dataset

def load_gazeta_dataset(n_samples=10):
    """
    Загружает датасет Gazeta для тестирования.
    Возвращает списки текстов и эталонных рефератов.
    """
    try:
        dataset = load_dataset('IlyaGusev/gazeta', split='test')
    except Exception as e:
        print(f"Ошибка загрузки датасета (проверьте интернет): {e}")
        return [], []

    texts = []
    references = []

    for i, item in enumerate(dataset):
        if i >= n_samples:
            break
        # Пропускаем слишком короткие тексты, где суммаризация не имеет смысла
        if len(item['text']) < 200:
            continue
        texts.append(item['text'])
        references.append(item['summary'])

    print(f'Загружено {len(texts)} примеров для теста')
    return texts, references