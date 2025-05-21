import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from collections import Counter

# ============================
# 모델 호출 & 인코더 로드
# ============================
model = load_model('C:/Users/swj03/Vite/slt/server/sign_model_fixed.h5')
encoder = joblib.load('C:/Users/swj03/Vite/slt/server/label_encoder.pkl')

# 라벨 순서 확인
print("현재 사용 중인 라벨 순서:", encoder.classes_)

# ============================
# MediaPipe 초기화
# ============================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# ============================
# 테스트할 수어 영상 경로
# ============================
cap = cv2.VideoCapture(0)  # 웹캠으로 실시간 입력 받기

sequence = []
predictions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    pose_result = pose.process(image)
    hands_result = hands.process(image)

    keypoints = []

    # 포즈: 어깨, 팔꿈치, 손목
    if pose_result.pose_landmarks:
        for idx in [11, 13, 15, 12, 14, 16]:
            lm = pose_result.pose_landmarks.landmark[idx]
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*18)

    # 손: 양손 최대 2개
    if hands_result.multi_hand_landmarks:
        for hand in hands_result.multi_hand_landmarks:
            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        if len(hands_result.multi_hand_landmarks) == 1:
            keypoints.extend([0]*63)
    else:
        keypoints.extend([0]*126)

    # keypoints 정보 출력
    print(f"[DEBUG] keypoints 길이: {len(keypoints)}, 평균: {np.mean(keypoints):.4f}, std: {np.std(keypoints):.4f}")

    if len(keypoints) != 144:
        continue

    sequence.append(keypoints)
    print(f"[DEBUG] 시퀀스 길이: {len(sequence)}")

    if len(sequence) == 30:
        input_data = np.expand_dims(sequence, axis=0)
        prediction = model.predict(input_data)[0]
        confidence = np.max(prediction)
        pred_label = encoder.inverse_transform([np.argmax(prediction)])[0]

        # 예측 출력
        print(f"[DEBUG] softmax score: {prediction}")
        print(f"[DEBUG] 예측: {pred_label}, 신뢰도: {confidence:.2f}")

        if confidence > 0.8:
            predictions.append(pred_label)
            print(f'예측 결과: {pred_label} ({confidence:.2f})')
        else:
            print(f'⚠️ 무시됨: {pred_label} ({confidence:.2f})')

        sequence = []

    # 웹캠 창 출력 (선택)
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ============================
# 최종 결과
# ============================
if predictions:
    print(f"\n전체 예측 리스트: {predictions}")
    final_word = Counter(predictions).most_common(1)[0][0]
    print(f"\n최종 번역 결과: {final_word}")
else:
    print("\n예측된 결과가 없습니다.")
