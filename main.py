import math
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

class Application(tk.Frame):
    disp_image_rate = 10

    def distance(obj, posA, posB):
        dx2 = pow(posA.x - posB.x, 2) * 1280
        dy2 = pow(posA.y - posB.y, 2) * 720
        dz2 = pow(posA.z - posB.z, 2) * 720
        return math.sqrt(dx2 + dy2 + dz2)
         
    def centerPoint(obj, posA, posB):
        centX = (posA.x + posB.x) / 2 * 1280
        centY = (posA.y + posB.y) / 2 * 720
        return [centX, centY]

    def __init__(self, master = None):
        super().__init__(master)
        self.pack()

        self.master.title("Mediapipe")
        self.master.geometry("1380x720")

        self.pinch = False
        self.points = []
        
        # Canvasの作成
        left_frame = tk.Frame(self, width=100, height=720)
        left_frame.grid(row=0, column=0, sticky=tk.N)
        right_frame = tk.Frame(self, width=1280, height=720)
        right_frame.grid(row=0, column=1, sticky=tk.N+tk.S)
        
        self.canvas = tk.Canvas(right_frame, width=1280, height=720)
        # Canvasにマウスイベント（左ボタンクリック）の追加
        self.canvas.bind('<Button-1>', self.canvas_click)
        # Canvasを配置
        self.canvas.pack(expand = True, fill = tk.BOTH)

        # ボタン
        def toggle_hand():
            if self.show_hand:
                self.show_hand = False
            else:
                self.show_hand = True
                self.show_holistic = False

        def toggle_holistic():
            if self.show_holistic:
                self.show_holistic = False
            else:
                self.show_holistic = True
                self.show_hand = False
        
        def toggle_line():
            if self.show_line:
                self.show_line = False
            else:
                self.show_line = True


        # ハンドボタン
        text = tk.StringVar(left_frame)
        text.set("Hand")
        button = tk.Button(left_frame, textvariable=text, command=toggle_hand)
        button.pack(padx=8, pady=2)

        # ホリスティックボタン
        text2 = tk.StringVar(left_frame)
        text2.set("Holistic")
        button2 = tk.Button(left_frame, textvariable=text2, command=toggle_holistic)
        button2.pack(padx=8, pady=2)

        # ホリスティックボタン
        text3 = tk.StringVar(left_frame)
        text3.set("Line")
        button3 = tk.Button(left_frame, textvariable=text3, command=toggle_line)
        button3.pack(padx=8, pady=2)

        # カメラをオープンする
        self.capture = cv2.VideoCapture(2)
        
        # Mediapipeオンオフ
        self.show_hand = False
        self.show_holistic = False
        self.show_line = True
        
        # 表示開始
        self.disp_image()

    def canvas_click(self, event):
        '''Canvasのマウスクリックイベント'''
        if self.disp_id is None:
            # 動画を表示
            self.disp_image()
        else:
            # 動画を停止
            self.after_cancel(self.disp_id)
            self.disp_id = None


    def disp_image(self):
        # ---------- フレーム画像の取得 ----------
        success, frame = self.capture.read()
        # frame.flags.writeable = False
        
        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv_image = cv2.flip(cv_image, 1)
        # cv_image.flags.writeable = False
        
        # ---------- Mediapipe ----------
        
        # Holistic
        if(self.show_holistic is True):
            with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:

                results = holistic.process(cv_image)
                mp_drawing.draw_landmarks(
                    cv_image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    cv_image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
                mp_drawing.draw_landmarks(
                    cv_image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_hand_landmarks_style())
                mp_drawing.draw_landmarks(
                    cv_image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_hand_landmarks_style())                
        # Hand
        elif(self.show_hand is True):
            with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
                results = hands.process(cv_image)

                if results.multi_hand_landmarks:
                    wrist = results.multi_hand_landmarks[0].landmark[0]
                    sum = results.multi_hand_landmarks[0].landmark[4]
                    index = results.multi_hand_landmarks[0].landmark[8]
                    dist = self.distance(wrist, sum)
                    pinch = self.distance(index, sum)
                    if(dist / 3 > pinch):
                        if(self.pinch == False):
                            self.points.append([])
                        self.pinch = True
                    else:
                        self.pinch = False
                    
                    if(self.pinch):
                        self.points[-1].append(self.centerPoint(index,sum))
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            cv_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

        # ---------- フレーム画像の取得 ----------
        # NumPyのndarrayからPillowのImageへ変換
        pil_image = Image.fromarray(cv_image)

        # キャンバスのサイズを取得
        # canvas_width = self.canvas.winfo_width()
        # canvas_height = self.canvas.winfo_height()
        canvas_width = 1280
        canvas_height = 720

        # 画像のアスペクト比（縦横比）を崩さずに指定したサイズ（キャンバスのサイズ）全体に画像をリサイズする
        pil_image = ImageOps.pad(pil_image, (canvas_width, canvas_height))

        # PIL.ImageからPhotoImageへ変換する
        self.photo_image = ImageTk.PhotoImage(image=pil_image)

        # 画像の描画
        self.canvas.create_image(
                canvas_width / 2,
                canvas_height / 2,
                image=self.photo_image)
        
        # 線の描画
        if(self.show_line and len(self.points) > 0):
            for listIndex in range(len(self.points)):
                if(len(self.points[listIndex]) >= 2):
                    for pointIndex in range(len(self.points[listIndex])-1):
                        self.canvas.create_line(
                            self.points[listIndex][pointIndex][0],
                            self.points[listIndex][pointIndex][1],
                            self.points[listIndex][pointIndex + 1][0],
                            self.points[listIndex][pointIndex + 1][1],
                            fill = "Red",
                            width = 6
                            )

        # disp_image()を10msec後に実行する
        self.disp_id = self.after(self.disp_image_rate, self.disp_image)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Frame Example")

    app = Application(master = root)
    app.mainloop()