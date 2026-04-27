import pymorphy3
import re
from ruwordnet import RuWordNet
from typing import List, Set, Dict
import nltk
from nltk.corpus import stopwords

morph = pymorphy3.MorphAnalyzer()
wn = RuWordNet()
stopwords = set(stopwords.words('russian'))

def preprocess_text(text) -> List[str]:
    text = text.lower().replace('ё', 'е')
    text = re.sub(r'[^а-яёa-z0-9\s\-]', '', text, flags=re.IGNORECASE)
    tokens = text.split()
    tokens = [token for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if token not in stopwords]
    for i in range(len(tokens)):
        parsed = morph.parse(tokens[i])[0]
        tokens[i] = parsed.normal_form
    return tokens

def get_word_senses_info(word) -> Dict:
    result = {
        'word': word,
        'senses': []
    }
    senses = wn.get_senses(word)
    for sense in senses:
        synset = sense.synset

        lemma =sense.name.lower()
        part_of_speech = synset.id.split('-')[-1] if '-' in synset.id else '?'
        definition = synset.title.lower()
        synonyms = [s.name.lower() for s in synset.senses if s.name != sense.name]
        hypernyms = [h.title.lower() for h in synset.hypernyms]
        hyponyms = [h.title.lower() for h in synset.hyponyms]

        sense_info = {
            'lemma': lemma,
            'part_of_speech': part_of_speech,
            'definition':definition ,
            'synonyms': synonyms,
            'hypernyms': hypernyms,
            'hyponyms': hyponyms
        }
        result['senses'].append(sense_info)
    return result

def print_wsd_result(context: str, target_word: str, target_lemma: str, context_lemmas: set, best_sense: dict) -> None:
    print(f"\nКонтекст: {context}")
    print(f"Целевое слово: {target_word} (лемма: {target_lemma})")
    print(f"Леммы контекста: {context_lemmas}")

    if best_sense:
        print(f"Выбранное значение:")
        print(f"  Лемма: {best_sense['lemma']}")
        print(f"  Часть речи: {best_sense['part_of_speech']}")
        print(f"  Определение: {best_sense['definition']}")
        print(f"  Синонимы: {', '.join(best_sense['synonyms']) if best_sense['synonyms'] else '—'}")
        print(f"  Гиперонимы: {', '.join(best_sense['hypernyms']) if best_sense['hypernyms'] else '—'}")
        print(f"  Гипонимы: {', '.join(best_sense['hyponyms']) if best_sense['hyponyms'] else '—'}")
    else:
        print("Не удалось определить значение.")

def lesk_wsd(context, target_word) -> Dict:
    target_lemma = morph.parse(target_word)[0].normal_form
    senses_info = get_word_senses_info(target_lemma)
    context_lemmas = set(preprocess_text(context))

    best_sense = None
    best_score = -1

    # Для каждого значения вычисляем пересечение с контекстом
    for sense in senses_info['senses']:
        # Берём глоссу (определение) и предобрабатываем её
        gloss = sense['definition']
        gloss_lemmas = set(preprocess_text(gloss))

        # добавляем глоссы
        for hyp in sense.get('hypernyms', []):
            # Для каждого гиперонима получаем его глоссу
            hyp_senses = wn.get_senses(hyp)  # восстанавливаем пробелы
            for h_sense in hyp_senses:
                h_gloss = h_sense.synset.title
                gloss_lemmas.update(preprocess_text(h_gloss))

        # Считаем количество общих лемм (размер пересечения)
        overlap = len(context_lemmas & gloss_lemmas)
        if overlap > best_score:
            best_score = overlap
            best_sense = sense
    if best_score == 0 and senses_info['senses']:
        best_sense = senses_info['senses'][0]

    print_wsd_result(context, target_word, target_lemma, context_lemmas, best_sense)

    return best_sense

def main():
    examples = [
        # 1. Оценка (школьная отметка vs мнение/ценность)
        ("Я буду сдавать лабораторные работы вовремя и получу хорошую оценку за экзамен.", "оценка"),
        ("Эксперт дал высокую оценку новому фильму.", "оценка"),

        # 2. Замок (строение vs запорное устройство)
        ("Туристы фотографировали древний замок на вершине холма.", "замок"),
        ("Не забудь закрыть дверь на замок.", "замок"),

        # 3. Ключ (инструмент vs источник vs музыкальный знак)
        ("Он потерял ключ от квартиры.", "ключ"),
        ("Вода бьёт из горного ключа.", "ключ"),
        ("Скрипичный ключ в нотах выглядит красиво.", "ключ"),

        # 4. Ручка (часть тела vs канцелярская принадлежность vs дверная ручка)
        ("У ребёнка болит ручка после падения.", "ручка"),
        ("Купи новую шариковую ручку в магазине.", "ручка"),
        ("Потяни за дверную ручку, чтобы открыть.", "ручка"),

        # 5. Коса (причёска vs сельхозорудие vs географическая форма)
        ("У неё длинная русая коса.", "коса"),
        ("Фермер точит косу, чтобы косить траву.", "коса"),
        ("Песчаная коса отделяла море от озера.", "коса"),

        # 6. Бокс (спорт vs коробка передач vs коробка для хранения)
        ("Он занимается боксом в спортивном клубе.", "бокс"),
        ("Автомобиль с автоматическим боксом удобен в городе.", "бокс"),
        ("Поставь инструменты в бокс для хранения.", "бокс"),

        # 7. Лук (растение vs оружие)
        ("Мама добавила зелёный лук в салат.", "лук"),
        ("Охотник натянул тетиву лука.", "лук"),

        # 8. Стекло (материал vs глагол в прош. вр. – стёк)
        ("Окно разбито, стекло валяется на полу.", "стекло"),
        ("Вода стекло с крыши после дождя.", "стекло"),   # омонимичная форма слова "стечь"

        # 9. Мир (отсутствие войны vs Вселенная)
        ("На Земле должен быть мир во всём мире.", "мир"),
        ("Учёные изучают тайны микромира.", "мир"),

        # 10. Лавка (магазин vs скамейка)
        ("Бабушка купила хлеб в угловой лавке.", "лавка"),
        ("Старики сидели на лавке у подъезда.", "лавка"),
    ]

    for context, target in examples:
        lesk_wsd(context, target)



if __name__ == "__main__":
    main()