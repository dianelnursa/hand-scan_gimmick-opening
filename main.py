import cv2                          # Import pustaka OpenCV untuk pengolahan gambar dan video
import mediapipe as mp              # Import pustaka MediaPipe untuk deteksi tangan
import numpy as np                  # Import NumPy untuk manipulasi array (opsional)

# Load gambar template tangan dengan alpha channel (transparansi)
template = cv2.imread('handscan.png', cv2.IMREAD_UNCHANGED)
template_h, template_w = template.shape[:2]  # Simpan tinggi dan lebar gambar template

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Inisialisasi modul deteksi tangan dari MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,        # False = gunakan untuk video streaming, bukan gambar statis
    max_num_hands=1,                # Deteksi maksimal satu tangan
    min_detection_confidence=0.9    # Ambang batas deteksi minimum 90%
)

# Fungsi untuk menempelkan gambar transparan (template tangan) ke frame video
def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    alpha_overlay = img_overlay[:, :, 3] / 255.0      # Ekstrak channel alpha (transparansi) dan normalisasi ke 0-1
    alpha_background = 1.0 - alpha_overlay            # Ambil nilai transparansi untuk background
    for c in range(0, 3):  # Untuk masing-masing channel warna (B, G, R)
        img[y:y+img_overlay.shape[0], x:x+img_overlay.shape[1], c] = (
            alpha_overlay * img_overlay[:, :, c] +    # Warna dari gambar template
            alpha_background * img[y:y+img_overlay.shape[0], x:x+img_overlay.shape[1], c]  # Warna background asli
        )

video_playing = False  # Flag penanda apakah video sedang diputar

# Mulai membaca frame dari kamera secara terus-menerus
while cap.isOpened():
    ret, frame = cap.read()  # Ambil frame dari kamera
    if not ret:
        break  # Keluar jika gagal membaca frame

    frame = cv2.flip(frame, 1)  # Mirror efek agar tampilan sesuai arah tangan kita
    h, w, _ = frame.shape  # Ambil ukuran frame

    # Hitung posisi tengah untuk meletakkan gambar template tangan
    center_x, center_y = w // 2 - template_w // 2, h // 2 - template_h // 2

    # Tempelkan template tangan ke atas frame
    overlay_image_alpha(frame, template, (center_x, center_y))

    # Konversi warna frame dari BGR ke RGB (karena MediaPipe pakai RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)  # Jalankan deteksi tangan pada frame

    should_play_video = False  # Flag untuk menentukan apakah video boleh diputar

    # Jika tangan terdeteksi dan ada info apakah tangan kiri/kanan
    if results.multi_hand_landmarks and results.multi_handedness:
        # Ambil label tangan ("Left" atau "Right")
        hand_label = results.multi_handedness[0].classification[0].label

        if hand_label == "Left":  # Filter: hanya tangan kiri yang valid
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Gambar landmark tangan

            # Landmark untuk pergelangan (wrist) dan tengah telapak (MCP jari tengah)
            wrist = hand_landmarks.landmark[0]
            mcp = hand_landmarks.landmark[9]

            # ID titik-titik ujung jari dan PIP (bengkokannya)
            tips = [4, 8, 12, 16, 20]       # Ujung jari: ibu jari - kelingking
            pips = [2, 6, 10, 14, 18]       # Sendi bawah jari (PIP)

            fingers_extended = 0            # Hitung jari yang terbuka
            all_inside_template = True      # Apakah semua ujung jari berada dalam area template?
            facing_camera = True            # Apakah tangan menghadap kamera?

            # Cek untuk semua jari
            for tip_id, pip_id in zip(tips, pips):
                tip = hand_landmarks.landmark[tip_id]
                pip = hand_landmarks.landmark[pip_id]

                # Konversi koordinat relatif ke piksel
                x_tip, y_tip = int(tip.x * w), int(tip.y * h)

                # Cek apakah ujung jari di dalam area gambar template
                if not (center_x <= x_tip <= center_x + template_w and
                        center_y <= y_tip <= center_y + template_h):
                    all_inside_template = False
                    break

                # Cek apakah jari terbuka (tip lebih tinggi dari pip)
                if tip.y < pip.y:
                    fingers_extended += 1

            # Cek apakah tangan menghadap kamera (z titik tengah telapak lebih dalam dari z pergelangan)
            if mcp.z - wrist.z < -0.05:
                facing_camera = True
            else:
                facing_camera = False

            # Semua syarat: 5 jari terbuka, berada dalam template, tangan menghadap kamera
            if fingers_extended == 5 and all_inside_template and facing_camera:
                should_play_video = True

    # Jika semua syarat terpenuhi dan video belum diputar
    if should_play_video and not video_playing:
        video_playing = True
        video = cv2.VideoCapture('play.mp4')  # Buka file video
        while video.isOpened():
            ret_vid, frame_vid = video.read()
            if not ret_vid:
                break  # Hentikan jika video selesai
            cv2.imshow('Hand Scanner', frame_vid)  # Tampilkan frame video
            if cv2.waitKey(30) & 0xFF == 27:  # Tekan ESC untuk keluar
                break
        video.release()  # Tutup video
        video_playing = False  # Reset status video

    cv2.imshow('Hand Scanner', frame)  # Tampilkan frame dari kamera
    if cv2.waitKey(5) & 0xFF == 27:  # Tekan ESC untuk keluar
        break

# Tutup kamera dan semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()
