import cv2
import mediapipe as mp
import numpy as np
import random

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

class Shape:
    def __init__(self, kind, color):
        self.kind = kind
        self.color = color
        self.size = 60
        self.pos = np.array([random.randint(100, 500), 0])
        self.speed = 5  # dapat di custom
        self.held = False
        self.matched = False   

    def draw(self, frame):
        x, y = self.pos
        if self.kind == 'square':
            cv2.rectangle(frame, (x, y), (x + self.size, y + self.size), self.color, -1)
        elif self.kind == 'circle':
            cv2.circle(frame, (x + self.size // 2, y + self.size // 2), self.size // 2, self.color, -1)
        elif self.kind == 'triangle':
            pts = np.array([[x, y + self.size], [x + self.size // 2, y], [x + self.size, y + self.size]], np.int32)
            cv2.drawContours(frame, [pts], 0, self.color, -1)

    def update(self):
        if not self.held and not self.matched:
            self.pos[1] += self.speed

    def is_point_inside(self, px, py):
        x, y = self.pos
        return x < px < x + self.size and y < py < y + self.size

    def match_to_slot(self, slot):
        x, y = self.pos
        sx, sy = slot.pos
        if abs(x - sx) < 10 and abs(y - sy) < 10:
            if self.kind == slot.kind:
                self.matched = True
                return True
        return False


class Slot:
    def __init__(self, kind, color, position):
        self.kind = kind
        self.color = color
        self.pos = position

    def draw(self, frame):
        x, y = self.pos
        if self.kind == 'square':
            cv2.rectangle(frame, (x, y), (x + 60, y + 60), self.color, 2)
        elif self.kind == 'circle':
            cv2.circle(frame, (x + 30, y + 30), 30, self.color, 2)
        elif self.kind == 'triangle':
            pts = np.array([[x, y + 60], [x + 30, y], [x + 60, y + 60]], np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=self.color, thickness=2)


cap = cv2.VideoCapture(1)   # 0: default webcam 1: kamera eksternal
shapes = [Shape(random.choice(['square', 'circle', 'triangle']),
                random.choice([(255, 0, 0), (0, 255, 0), (0, 128, 255)]))]

slots = [
    Slot('square', (0, 0, 0), (500, 100)),   
    Slot('circle', (0, 0, 0), (500, 200)),  
    Slot('triangle', (0, 0, 0), (500, 300)) 
]

grabbed_shape = None
score = 0
fall_speed = 3  # dapat dicustom

def is_hand_closed(hand_landmarks):
    tip_ids = [8, 12, 16, 20]
    fingers = []
    for tip_id in tip_ids:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers) == 0  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_pos = None
    closed = False

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            cx = int(handLms.landmark[9].x * w)
            cy = int(handLms.landmark[9].y * h)
            hand_pos = np.array([cx, cy])
            closed = is_hand_closed(handLms)

    for shape in shapes:
        shape.speed = fall_speed   
        shape.update()
        shape.draw(frame)

    for slot in slots:
        slot.draw(frame)

    if hand_pos is not None:
        if closed:
            if grabbed_shape is None:
                for shape in reversed(shapes):   
                    if shape.is_point_inside(*hand_pos):
                        shape.held = True
                        grabbed_shape = shape
                        break
            elif grabbed_shape:
                grabbed_shape.pos = hand_pos - grabbed_shape.size // 2
        else:
            if grabbed_shape:
                grabbed_shape.held = False
                grabbed_shape = None

    # Match shape with slot
    if grabbed_shape:
        for slot in slots:
            if grabbed_shape.match_to_slot(slot):
                score += 1
                shapes.remove(grabbed_shape)
                grabbed_shape = None
                break

    cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, 'Ambil: Tutup tangan ✊ | Geser: Gerakkan tangan | Lepas: Buka tangan ✋',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if random.randint(1, 100) == 1:
        shapes.append(Shape(random.choice(['square', 'circle', 'triangle']),
                            random.choice([(255, 0, 0), (0, 255, 0), (0, 128, 255)])))

    cv2.imshow("Gesture Shape Puzzle", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
