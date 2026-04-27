import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import kagglehub
import warnings

warnings.filterwarnings('ignore')

# --- Константы ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_SIZE = 50000
MAX_LENGTH = 128
MODEL_NAME = "DeepPavlov/rubert-base-cased"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
TFIDF_MAX_FEATURES = 50000

def load_and_prepare_data():
    # Скачивание датасета
    path = kagglehub.dataset_download("kyakovlev/yandex-geo-reviews-dataset-2023")
    file_path = os.path.join(path, "geo-reviews-dataset-2023.csv")

    # Чтение только нужных колонок
    df = pd.read_csv(file_path, usecols=["text", "rating"])

    # Очистка
    df = df.dropna(subset=["text", "rating"])
    df = df[df["text"].str.len() > 10]

    # Приведение рейтинга к шкале 1-10 (если исходная 1-5, то умножаем на 2)
    # В датасете Яндекса рейтинг обычно 1-5.
    # Если в датасете уже 1-10, то умножение не нужно.
    # Исходя из оригинального кода (rating * 2), предполагаем исходную шкалу 1-5.
    df["score"] = df["rating"] * 2

    # Сэмплирование для ускорения экспериментов
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    return df[["text", "score"]].reset_index(drop=True)

df = load_and_prepare_data()
print(f"Dataset shape: {df.shape}")

# --- Базовая модель: TF-IDF + Ridge ---
def train_tfidf_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["score"], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    preds_clipped = np.clip(preds, 1, 10)

    rmse = np.sqrt(mean_squared_error(y_test, preds_clipped))
    print(f"TF-IDF Ridge RMSE: {rmse:.4f}")

    return vectorizer, model

vectorizer, tfidf_model = train_tfidf_model(df)

# --- Продвинутая модель: RuBERT Regressor ---
def prepare_bert_datasets(df):
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Нормализуем label в [0, 1] для стабильности обучения нейросети,
    # но метрики будем считать в исходной шкале [1, 10]
    train_df["label"] = (train_df["score"] - 1) / 9.0
    test_df["label"] = (test_df["score"] - 1) / 9.0

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    # Удаляем лишние колонки, оставляем только тензоры
    cols_to_keep = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=cols_to_keep)
    test_dataset.set_format(type="torch", columns=cols_to_keep)

    return train_dataset, test_dataset, tokenizer

class BertRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        x = self.dropout(cls_output)
        logits = self.regressor(x).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Обратное преобразование из [0, 1] в [1, 10] для расчета метрики
    predictions_scaled = predictions.squeeze() * 9.0 + 1.0
    labels_scaled = labels.squeeze() * 9.0 + 1.0

    rmse = np.sqrt(mean_squared_error(labels_scaled, predictions_scaled))
    return {"rmse": rmse}

def train_bert_model(df):
    train_dataset, test_dataset, tokenizer = prepare_bert_datasets(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertRegressor(MODEL_NAME).to(device)

    training_args = TrainingArguments(
        output_dir="./bert_rating_model",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        logging_steps=100,
        fp16=torch.cuda.is_available(),  # Включаем fp16 только если есть CUDA
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer, tokenizer

bert_trainer, bert_tokenizer = train_bert_model(df)

# --- Инференс и Сравнение ---
def predict_tfidf(texts, vectorizer, model):
    vec = vectorizer.transform(texts)
    preds = model.predict(vec)
    return np.clip(preds, 1, 10)

def predict_bert(texts, trainer, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    model = trainer.model
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"].cpu().numpy()

    # Преобразование из [0, 1] обратно в [1, 10]
    scores = logits * 9.0 + 1.0
    return np.clip(scores, 1, 10)

# Тестовые примеры
test_texts = [
    "Потрясающе! Обязательно вернусь снова и посоветую всем друзьям.",
    "Отвратительно. Ждал заказ час, принесли холодным. Персонал хамит.",
    "Средне. Интерьер симпатичный, но еда совершенно обычная, ничего особенного.",
    "Ну спасибо, «порадовали» так порадовали... Больше ни ногой сюда.",
    "Не могу сказать, что мне не понравилось, но и восторга не вызвало.",
    "Раньше было гораздо лучше, сейчас качество сильно упало.",
    "Круто!",
    "Ужас.",
    "Очень дорого для такого посредственного уровня исполнения."
]

# Получаем предсказания
tfidf_scores = predict_tfidf(test_texts, vectorizer, tfidf_model)

bert_scores = predict_bert(test_texts, bert_trainer, bert_tokenizer)

# Вывод результатов
print(f"{'Text':<60} | {'TF-IDF':<6} | {'BERT':<6}")
print("-" * 90)
for text, t_score, b_score in zip(test_texts, tfidf_scores, bert_scores):
    # Обрезаем текст для красивого вывода
    short_text = text[:57] + "..." if len(text) > 60 else text
    print(f"{short_text:<60} | {t_score:<6.2f} | {b_score:<6.2f}")