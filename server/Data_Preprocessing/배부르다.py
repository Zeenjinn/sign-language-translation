import cv2
import numpy as np
import mediapipe as mp
import os

# ==============================
# 설정
# ==============================
label = '배부르다'  # 저장할 수어 이름

# 현재 파일 경로 기준으로 데이터 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, 'sign_data')
SLV_PATH = os.path.join(CURRENT_DIR, 'SLV')

VIDEO_LIST = [
    os.path.join(SLV_PATH, '배부르다1.mp4'),
    os.path.join(SLV_PATH, '배부르다2.mp4'),
    os.path.join(SLV_PATH, '배부르다3.mp4'),
    os.path.join(SLV_PATH, '배부르다4.mp4')
]

# 저장 폴더 생성
save_dir = os.path.join(DATA_PATH, label)
os.makedirs(save_dir, exist_ok=True)

existing = os.listdir(save_dir)
saved_count = len([f for f in existing if f.endswith('.npy')])

# ==============================
# MediaPipe 초기화
# ==============================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# ==============================
# 영상 목록 처리
# ==============================
for video_path in VIDEO_LIST:
    if not os.path.exists(video_path):
        print(f"파일 없음: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    sequence = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[INFO] 영상 처리 시작: {video_path} | 총 프레임: {total_frames}")
    valid_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            continue

        image.flags.writeable = False
        pose_result = pose.process(image)
        hands_result = hands.process(image)

        keypoints = []

        # 포즈
        if pose_result.pose_landmarks:
            for idx in [11, 13, 15, 12, 14, 16]:
                lm = pose_result.pose_landmarks.landmark[idx]
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0] * 18)

        # 손
        if hands_result.multi_hand_landmarks:
            detected_hands = hands_result.multi_hand_landmarks[:2]
            for hand in detected_hands:
                for lm in hand.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            if len(detected_hands) == 1:
                keypoints.extend([0] * 63)
        else:
            keypoints.extend([0] * 126)

        if len(keypoints) != 144:
            continue

        sequence.append(keypoints)
        valid_count += 1

        if len(sequence) == 30:
            npy_path = os.path.join(save_dir, f'{saved_count}.npy')
            np.save(npy_path, np.array(sequence))
            print(f"저장 완료: {saved_count}.npy")
            sequence = []
            saved_count += 1

    cap.release()
    print(f"[완료] {video_path} | 처리 프레임 수: {total_frames} | 유효 프레임 수: {valid_count}")

print(f"\n[전체 완료] 최종 저장된 시퀀스 수: {saved_count}")
cv2.destroyAllWindows()
