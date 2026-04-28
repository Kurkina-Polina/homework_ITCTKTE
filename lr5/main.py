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
    text = text.lower()
    text = re.sub(r'[^а-яa-z0-9\s\-]', '', text, flags=re.IGNORECASE)
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
    context_lemmas.discard(target_lemma)

    best_sense = None
    best_score = -1

    # Для каждого значения вычисляем пересечение с контекстом
    for sense in senses_info['senses']:
        score = 0

        # Глосса (определение)
        gloss_lemmas = set(preprocess_text(sense['definition']))
        # Исключаем само целевое слово из глоссы, если оно там есть, чтобы не накручивать счетчик
        gloss_lemmas.discard(target_lemma)
        overlap_gloss = len(context_lemmas & gloss_lemmas)
        score += overlap_gloss * 2  # Вес 2 за совпадение с определением

        synonym_lemmas = set()
        for syn in sense.get('synonyms', []):
            syn_tokens = preprocess_text(syn)
            synonym_lemmas.update(syn_tokens)
        overlap_syn = len(context_lemmas & synonym_lemmas)
        score += overlap_syn * 1.5

        # Гиперонимы (родовые понятия)
        hypernym_lemmas = set()
        for hyp_title in sense.get('hypernyms', []):
            hyp_tokens = preprocess_text(hyp_title)
            hypernym_lemmas.update(hyp_tokens)
        overlap_hyper = len(context_lemmas & hypernym_lemmas)
        score += overlap_hyper * 1.0

        # Гипонимы (видовые понятия)
        hyponym_lemmas = set()
        for hypo_title in sense.get('hyponyms', []):
            hypo_tokens = preprocess_text(hypo_title)
            hyponym_lemmas.update(hypo_tokens)
        overlap_hypo = len(context_lemmas & hyponym_lemmas)
        score += overlap_hypo * 0.5

        if score > best_score:
            best_score = score
            best_sense = sense

    if best_score == 0 and senses_info['senses']:
        best_sense = senses_info['senses'][0]

    print_wsd_result(context, target_word, target_lemma, context_lemmas, best_sense)

    return best_sense

def main():
    examples = [
        # 1. Оценка (школьная отметка vs мнение/ценность)
        ("Ученик получил хорошую учебную оценку и даже пятёрку за ответ.", "оценка"),
        ("Эксперт дал свою оценку, высказал мнение и повысил рейтинг фильму.", "оценка"),

        # 2. Замок (строение vs запорное устройство)
        ("Древний замок выглядел как неприступная крепость феодала.", "замок"),
        ("Чтобы открыть дверь, нужно отпереть запорный замок.", "замок"),

        # 3. Ключ (инструмент vs источник vs музыкальный знак)
        ("Он вставил ключ в дверной замок, чтобы войти.", "ключ"),
        ("Вода бьёт из горного ключа.", "ключ"),
        ("Скрипичный ключ — это важный музыкальный знак в нотах.", "ключ"),

        # 4. Ручка (часть тела vs канцелярская принадлежность vs дверная ручка)
        ("У ребёнка болит ручка после падения, теперь эту руку нужно забинтовать.", "ручка"),
        ("Для письма я использую шариковую ручку и другую пишущую принадлежность.", "ручка"),
        ("Потяни за дверную ручку, чтобы открыть.", "ручка"),

        # 5. Коса (причёска vs сельхозорудие vs географическая форма)
        ("У неё длинная русая коса.", "коса"),
        ("Песчаная коса отделяла море от озера.", "коса"),

        # 6. Бокс (спорт vs коробка передач vs коробка для хранения)
        ("Бокс - опасный вид спорта", "бокс"),
        ("Самое сложное на экзамене - парковка в бокс", "бокс"),
        ("Его палата находится в инфекционном боксе.", "бокс"),

        # 7. Лук (растение vs оружие)
        ("Мама добавила зелёный лук в салат.", "лук"),
        ("Из оружия ему больше всего нравится лук.", "лук"),

        # 9. Мир (отсутствие войны vs Вселенная)
        ("Поскорее бы кончилась война и настал мир.", "мир"),
        ("Учёные изучают тайны микромира.", "мир"),

        # 10. Лавка (магазин vs скамейка)
        ("Бабушка купила хлеб в торговой лавке.", "лавка"),
        ("Старики сидели на лавке у подъезда, потому что других скамеек нет.", "лавка"),
    ]

    for context, target in examples:
        lesk_wsd(context, target)



if __name__ == "__main__":
    main()