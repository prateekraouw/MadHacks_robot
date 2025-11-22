import genesis as gs
import numpy as np
import cv2
import time
import mediapipe as mp
import sys

# --- CONFIGURATION ---
URDF_PATH = "my_robot.urdf"
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# MOVEMENT SETTINGS
SCALE_X = 0.0008
SCALE_Y = 0.0008
MOVE_COOLDOWN = 4.0  # Time to freeze after moving
LOCK_TIME = 1.0      # Time to hold object still before locking

# Red Color Settings
LOWER_RED1 = np.array([0, 120, 70])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 120, 70])
UPPER_RED2 = np.array([180, 255, 255])

class State:
    SEARCHING = "SEARCHING"
    MOVING = "MOVING (LOCKED)"

def boxes_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

def main():
    # --- 1. INIT GENESIS ---
    print("⏳ Initializing Genesis...")
    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())
    
    try:
        robot = scene.add_entity(gs.morphs.URDF(file=URDF_PATH, fixed=True))
    except Exception as e:
        print(f"❌ URDF Error: {e}")
        return

    target_visual = scene.add_entity(
        gs.morphs.Sphere(radius=0.03, pos=(0.2, 0.0, 0.1), fixed=True),
        surface=gs.surfaces.Rough(color=(1.0, 0.0, 0.0))
    )
    scene.build()

    # Find End Effector
    end_effector = None
    for name in ['gripper_link', 'wrist_link', 'hand', 'tip', 'link_4']:
        try:
            end_effector = robot.get_link(name)
            break
        except: continue
    if not end_effector: end_effector = robot.links[-1]

    # --- 2. INIT VISION ---
    print(f"📷 Opening Camera {CAMERA_ID}...")
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(CAMERA_ID)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    time.sleep(1)

    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

    # --- VARIABLES ---
    current_state = State.SEARCHING
    last_move_time = 0
    target_lock_start = 0
    potential_target = None 
    
    # ROBOT STATE MEMORY (Prevents flipping)
    # We initialize it to the starting position
    active_target_pos = np.array([0.2, 0.0, 0.1])
    current_qpos = robot.get_qpos() # Store current angles

    print("✅ SYSTEM READY.")

    running = True
    while running:
        # A. VISION PROCESSING
        ret, frame = cap.read()
        if not ret: 
            try: scene.step()
            except: running = False
            continue
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect Humans
        human_boxes = []
        results = face_detection.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                bx, by = int(bbox.xmin * w), int(bbox.ymin * h)
                bw, bh = int(bbox.width * w), int(bbox.height * h)
                human_boxes.append((bx, by, bw, bh))
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)

        # Detect Red Target
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1) + cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_valid_target = False
        curr_cx, curr_cy = 0, 0

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                x, y, cw, ch = cv2.boundingRect(c)
                curr_cx, curr_cy = x + cw//2, y + ch//2
                
                target_rect = (x, y, cw, ch)
                if not any(boxes_overlap(target_rect, hb) for hb in human_boxes):
                    found_valid_target = True
                    color = (0, 255, 255) if current_state == State.SEARCHING else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+cw, y+ch), color, 2)
                    cv2.circle(frame, (curr_cx, curr_cy), 5, color, -1)

        # B. STATE MACHINE LOGIC
        now = time.time()

        if current_state == State.MOVING:
            time_left = int(MOVE_COOLDOWN - (now - last_move_time))
            cv2.putText(frame, f"LOCKED: {time_left}s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if now - last_move_time > MOVE_COOLDOWN:
                current_state = State.SEARCHING
                print("🔓 Search Resumed.")

        elif current_state == State.SEARCHING:
            if found_valid_target:
                if potential_target is None:
                    potential_target = (curr_cx, curr_cy)
                    target_lock_start = now
                else:
                    dist = np.hypot(curr_cx - potential_target[0], curr_cy - potential_target[1])
                    if dist < 30: 
                        # --- TRIGGER MOVEMENT (HAPPENS ONCE) ---
                        if now - target_lock_start > LOCK_TIME:
                            current_state = State.MOVING
                            last_move_time = now
                            
                            # 1. Calculate Coordinate
                            dx = curr_cx - (w // 2)
                            dy = curr_cy - (h // 2)
                            robot_x = 0.3 - (dy * SCALE_Y)
                            robot_y = -(dx * SCALE_X)
                            robot_z = 0.10
                            
                            # Clamp
                            robot_x = max(0.1, min(0.5, robot_x))
                            robot_y = max(-0.4, min(0.4, robot_y))
                            
                            active_target_pos = np.array([robot_x, robot_y, robot_z])
                            
                            print(f"🚀 CALCULATING IK FOR: {active_target_pos}")
                            
                            # 2. SOLVE IK ONCE AND STORE RESULT
                            try:
                                new_qpos = robot.inverse_kinematics(
                                    link=end_effector,
                                    pos=active_target_pos,
                                    quat=np.array([0, 1, 0, 0])
                                )
                                # Update the "Holding Pose"
                                current_qpos = new_qpos 
                            except:
                                print("⚠️ Target unreachable, holding previous position.")

                            # Flash screen
                            cv2.rectangle(frame, (0,0), (w,h), (0,255,0), 10)
                    else:
                        potential_target = (curr_cx, curr_cy)
                        target_lock_start = now
            else:
                potential_target = None

        # C. GENESIS UPDATE
        
        # Move visual red ball
        target_visual.set_pos(active_target_pos)
        
        # Apply the stored angles (STABILITY FIX)
        # We set the robot to 'current_qpos' every frame. 
        # Since 'current_qpos' only changes when we lock a new target, no flipping occurs.
        robot.set_qpos(current_qpos)
        
        try:
            scene.step()
        except gs.GenesisException:
            print("👋 Genesis Viewer Closed.")
            running = False

        # D. GUI
        cv2.putText(frame, f"STATE: {current_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Robot Eye", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()

if __name__ == "__main__":
    main()