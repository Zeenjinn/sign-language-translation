# ✅ LSTM.py
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# 데이터 로드 함수 (수정됨: 1단계 폴더 구조에 맞춤)
def load_data_flat(data_path):
    sequences, sequence_labels = [], []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            if file.endswith('.npy'):
                file_path = os.path.join(label_path, file)
                sequence = np.load(file_path)
                if sequence.shape == (30, 144):
                    sequences.append(sequence)
                    sequence_labels.append(label)
    return np.array(sequences), sequence_labels

# 모델 생성 함수
def build_model(input_shape, num_classes, num_units, num_layers=2, dropout_rate=0.5):
    model = Sequential()
    model.add(LSTM(num_units, return_sequences=True if num_layers > 1 else False, input_shape=input_shape))
    for i in range(1, num_layers):
        model.add(LSTM(num_units, return_sequences=(i < num_layers - 1)))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(num_units // 2, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 학습 및 평가 함수
def train_and_evaluate(X_train, y_train, X_test, y_test, model):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[early_stopping])
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='macro')
    recall = recall_score(y_true_classes, y_pred_classes, average='macro')
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
    return history, accuracy, precision, recall, f1

# 메인 실행 함수
def main():
    DATA_PATH = os.path.join('Data_Preprocessing', 'sign_data')
    X, y_labels = load_data_flat(DATA_PATH)
    
    if len(X) == 0:
        print("학습할 데이터가 없습니다. .npy 파일 경로를 확인하세요.")
        return

    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    y_cat = to_categorical(y)
    joblib.dump(le, 'label_encoder.pkl')
    print("라벨 인코더 저장 완료 → label_encoder.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)
    num_classes = len(np.unique(y_labels))
    unit_options = [128]
    num_layers_options = [2]
    dropout_options = [0.25]

    for num_units in unit_options:
        for num_layers in num_layers_options:
            for dropout_rate in dropout_options:
                print(f"\nTraining with {num_units} units, {num_layers} layers, dropout {dropout_rate}")
                model = build_model((30, 144), num_classes, num_units, num_layers, dropout_rate)
                history, accuracy, precision, recall, f1 = train_and_evaluate(X_train, y_train, X_test, y_test, model)
                print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
                model_name = f"sign_model_fixed.h5"
                model.save(model_name)
                print(f"모델 저장 완료 → {model_name}")

if __name__ == "__main__":
    main()
