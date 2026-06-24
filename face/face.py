import cv2
import mediapipe as mp

# MediaPipeの顔検出（Face Mesh）を設定
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Webカメラを起動 (0は内蔵カメラ)
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1, # 同時に検出する顔の数
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("カメラから映像を取得できませんでした。")
            break

        # 処理速度向上のため、画像を書き込み不可にする
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # 画面に描画するために再びBGRに戻す
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 顔の特徴点（ランドマーク）が検出された場合、画面にメッシュを描画
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # ウィンドウに映像を表示
        cv2.imshow('MediaPipe Face Mesh', image)

        # キーボードの「Escキー」を押すと終了
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
# Mac用のフリーズ対策（余韻を持たせてウインドウの破棄を確実にする）
for i in range(1, 5):
    cv2.waitKey(1)