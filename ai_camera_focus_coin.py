import cv2
import numpy as np
import simpleaudio as sa
import time
import os
import csv
from datetime import datetime

# ==============================
# アラート音生成
# ==============================
def create_beep(frequency=1000, duration=0.4, sample_rate=44100):
    """
    シンプルなビープ音を生成して WaveObject を返す
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(frequency * 2 * np.pi * t)
    audio = wave * (2**15 - 1) / np.max(np.abs(wave))
    audio = audio.astype(np.int16)
    return sa.WaveObject(audio, 1, 2, sample_rate)


beep_sound = create_beep()

# ==============================
# 集中度・サボり判定パラメータ
# ==============================
ATTENTION_MAX = 100
ATTENTION_MIN = 0
ATTENTION_FOCUS_INC = 4.0     # 集中中のときの回復速度（ポイント/秒）

# サボりタイプごとの減点速度（ポイント/秒）
PENALTY_AWAY = 10.0           # 離席（顔が映ってない）
PENALTY_OFFCENTER = 5.0       # ずっと画面端
PENALTY_FAR = 4.0             # 画面から遠い
PENALTY_RESTLESS = 2.0        # キョロキョロ

# 顔位置に関する閾値
OFFCENTER_X_RATIO = 0.25      # 横方向で中心から25%以上ズレたら「端」
OFFCENTER_Y_RATIO = 0.25      # 縦方向で中心から25%以上ズレたら「端」
MIN_FACE_AREA_RATIO = 0.04    # 顔面積/画面全体 がこれ未満なら「遠い」

# 動き（落ち着きなさ）の閾値
MOVEMENT_SPEED_THRESH = 0.25  # 顔中心の移動速度閾値（正規化）

# アラーム条件
ALARM_THRESHOLD = 30          # 集中度がこれ未満で
ALARM_MIN_BAD_SECONDS = 5     # 「悪い状態」がこの秒数続くと警告音

# ==============================
# コイン関連パラメータ
# ==============================
ATTENTION_COIN_THRESHOLD = 60     # この集中度以上のときだけコイン付与
COIN_FACTOR = 0.1                 # 集中度 × 時間(分) × 0.1

COIN_CSV_PATH = "focus_coins_log.csv"


# ==============================
# CSVから累計コインを読み込み
# ==============================
def load_global_coins(csv_path: str) -> float:
    """
    既存のCSVから最後の total_coins を読み取って返す。
    ファイルがない or 読み取れない場合は 0.0 を返す。
    """
    if not os.path.exists(csv_path):
        return 0.0

    last_row = None
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row

    if last_row is None:
        return 0.0

    try:
        return float(last_row.get("total_coins", 0.0))
    except ValueError:
        return 0.0


# ==============================
# CSVに今回セッションの結果を追記
# ==============================
def append_session_to_csv(csv_path: str, session_info: dict):
    """
    セッション結果をCSVに1行追記する。
    なければヘッダ付きで新規作成。
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "date",
            "session_start",
            "session_end",
            "session_minutes",
            "session_coins",
            "total_coins",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(session_info)


def main():
    # ==============================
    # 顔検出器（Haar Cascade）
    # ==============================
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # ==============================
    # カメラ起動
    # ==============================
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが開けませんでした。別の番号 (1, 2...) を試すか、接続を確認してください。")
        return

    attention = 100.0
    last_time = time.time()

    # 各サボり状態がどれくらい継続しているか（秒）
    away_time = 0.0
    offcenter_time = 0.0
    far_time = 0.0
    restless_time = 0.0

    # 直前フレームでの顔中心
    prev_face_cx = None
    prev_face_cy = None

    # アラーム用
    bad_time_accum = 0.0
    alarm_lock = False
    last_print = 0.0

    # ------------------------------
    # コイン関連
    # ------------------------------
    session_start_time = time.time()
    session_start_dt = datetime.now()

    # 既存CSVから累計コイン読み込み
    global_coins = load_global_coins(COIN_CSV_PATH)
    session_coins = 0.0

    # 何分目までコイン計算済みか（経過分）
    last_coin_minute = 0

    print("AIカメラβ2（詳細サボり検知＋コイン＋CSV保存・非表示モード）起動")
    print(f"これまでの累計コイン: {global_coins:.1f}")
    print("終了するにはターミナルで Ctrl + C を押してください。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレームが取得できませんでした。")
                break

            now = time.time()
            dt = now - last_time
            if dt <= 0:
                dt = 1e-3
            last_time = now

            h, w, _ = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 顔検出
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(60, 60),
            )

            # 状態フラグ初期化
            face_present = False
            is_offcenter = False
            is_far = False
            is_restless = False

            if len(faces) == 0:
                # 顔なし → 離席状態
                face_present = False
                away_time += dt
                offcenter_time = 0.0
                far_time = 0.0
                restless_time = 0.0
                prev_face_cx = None
                prev_face_cy = None
            else:
                face_present = True
                away_time = 0.0

                # 一番大きい顔を採用
                faces_sorted = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                (x, y, fw, fh) = faces_sorted[0]

                # 顔中心と大きさ
                face_cx = x + fw / 2
                face_cy = y + fh / 2
                face_area_ratio = (fw * fh) / (w * h)

                # 中心からのずれ量（0〜0.5くらい）
                offset_x = abs(face_cx - w / 2) / w
                offset_y = abs(face_cy - h / 2) / h

                # 画面端かどうか
                if offset_x > OFFCENTER_X_RATIO or offset_y > OFFCENTER_Y_RATIO:
                    is_offcenter = True
                    offcenter_time += dt
                else:
                    is_offcenter = False
                    offcenter_time = 0.0

                # 遠いかどうか
                if face_area_ratio < MIN_FACE_AREA_RATIO:
                    is_far = True
                    far_time += dt
                else:
                    is_far = False
                    far_time = 0.0

                # キョロキョロしているかどうか（顔中心の移動速度）
                if prev_face_cx is not None and prev_face_cy is not None:
                    dx = (face_cx - prev_face_cx) / w
                    dy = (face_cy - prev_face_cy) / h
                    speed = (dx**2 + dy**2) ** 0.5 / dt

                    if speed > MOVEMENT_SPEED_THRESH:
                        is_restless = True
                        restless_time += dt
                    else:
                        is_restless = False
                        restless_time = 0.0
                else:
                    is_restless = False
                    restless_time = 0.0

                prev_face_cx = face_cx
                prev_face_cy = face_cy

            # ==============================
            # 集中度の更新
            # ==============================
            if face_present and not is_offcenter and not is_far and not is_restless:
                # 理想状態：画面中央・近い・落ち着いている
                attention += ATTENTION_FOCUS_INC * dt
            else:
                # サボり状態ごとに減点（重複したら合算）
                penalty = 0.0
                if not face_present:
                    penalty += PENALTY_AWAY
                if is_offcenter:
                    penalty += PENALTY_OFFCENTER
                if is_far:
                    penalty += PENALTY_FAR
                if is_restless:
                    penalty += PENALTY_RESTLESS

                attention -= penalty * dt

            # 0〜100にクリップ
            attention = max(ATTENTION_MIN, min(ATTENTION_MAX, attention))

            # ==============================
            # コイン加算処理
            # ==============================
            elapsed_sec = now - session_start_time
            # 経過時間（分）を秒から換算し、小数切り捨て
            elapsed_min = int(elapsed_sec // 60)

            if elapsed_min > last_coin_minute:
                # 新たに経過した「フルの分」の数
                added_minutes = elapsed_min - last_coin_minute

                # 指定仕様：
                # 集中度が閾値以上のときだけ
                #   コイン += 集中度 × 時間(分) × 0.1
                if attention >= ATTENTION_COIN_THRESHOLD:
                    coins_added = attention * added_minutes * COIN_FACTOR
                    session_coins += coins_added
                    global_coins += coins_added

                last_coin_minute = elapsed_min

            # ==============================
            # アラーム用「悪い状態」継続時間
            # ==============================
            if (not face_present) or is_offcenter or is_far:
                bad_time_accum += dt
            else:
                bad_time_accum = 0.0
                alarm_lock = False

            # アラーム判定
            if (
                attention < ALARM_THRESHOLD
                and bad_time_accum >= ALARM_MIN_BAD_SECONDS
                and not alarm_lock
            ):
                print("⚠ 集中度低下！アラーム再生！")
                beep_sound.play()
                alarm_lock = True

            # ==============================
            # 状態テキスト
            # ==============================
            if not face_present:
                status = "離席"
                detail = f"離席継続: {away_time:.1f}s"
            else:
                tags = []
                if is_offcenter:
                    tags.append(f"画面端 {offcenter_time:.1f}s")
                if is_far:
                    tags.append(f"遠い {far_time:.1f}s")
                if is_restless:
                    tags.append(f"落ち着きなし {restless_time:.1f}s")

                if not tags:
                    status = "集中中"
                    detail = "良好"
                else:
                    status = "サボり気味"
                    detail = " / ".join(tags)

            # ==============================
            # ログ表示（1秒ごと）
            # ==============================
            if now - last_print > 1.0:
                print(
                    f"[状態] 集中度: {int(attention):3d} / {status} ({detail})"
                    f" / 経過: {elapsed_min}分"
                    f" / このセッション: {session_coins:.1f}コイン"
                    f" / 累計: {global_coins:.1f}コイン"
                )
                last_print = now

            # CPU使いすぎ防止
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nユーザーによって停止されました。")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        session_end_dt = datetime.now()
        total_minutes = int((time.time() - session_start_time) // 60)

        # セッション結果をCSVに記録
        session_info = {
            "date": session_start_dt.strftime("%Y-%m-%d"),
            "session_start": session_start_dt.strftime("%H:%M:%S"),
            "session_end": session_end_dt.strftime("%H:%M:%S"),
            "session_minutes": total_minutes,
            "session_coins": f"{session_coins:.1f}",
            "total_coins": f"{global_coins:.1f}",
        }
        append_session_to_csv(COIN_CSV_PATH, session_info)

        print(
            f"AIカメラβ2を終了しました。\n"
            f"このセッション獲得コイン: {session_coins:.1f}\n"
            f"累計コイン: {global_coins:.1f}\n"
            f"ログ: {COIN_CSV_PATH} に保存しました。"
        )


if __name__ == "__main__":
    main()
