from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import queue
from waitress import serve

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Setup Mediapipe and OpenCV
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define drawing specifications
drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=1)

#current_row_index = 2

# Queue for sending updates to the client
update_queue = queue.Queue()

# Function to process video frames and detect gaze direction
def process_frame(frame):
    global current_row_index

    # Resize frame to a smaller resolution
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    start = time.time()

    img_h , img_w, img_c = image.shape
    face_2d = []
    face_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append(([x, y, lm.z]))
            # Get 2d Coord
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            distortion_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
            # getting rotational of face
            rmat, jac = cv2.Rodrigues(rotation_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # here based on axis rot angle is calculated
            if y < -4:
                text = "Looking Left"
            elif y > 4:
                text = "Looking Right"
            elif x < -2:
                text = "Looking Down"
            elif x > 7:
                text = "Looking Up"
            else:
                text = "Forward"

            if text == "Looking Left":
                update_queue.put({"color": "left_window_green"})  # Put the new row index in the queue
            elif text == "Looking Right":
                update_queue.put({"color": "right_window_green"})
            elif text == "Looking Down":
                update_queue.put({"color": "bottom_window_green"})
            elif text == "Looking Up":
                update_queue.put({"color": "top_window_green"})
            else:
                update_queue.put({"color": None})


            #nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix,
            #                                                 distortion_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        #print("FPS: ", fps)
        #cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(image=image,
                                  landmark_list=face_landmarks,
                                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=drawing_spec,
                                  connection_drawing_spec=drawing_spec)

    # Encode frame as JPEG to stream via Flask
    ret, buffer = cv2.imencode('.jpg', image)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

ua_file_path = 'vocab/ukrainian_vocabulary.csv'
ua_df = pd.read_csv(ua_file_path, usecols=['Ukrainian Word', 'English Translation', 'Random Word 1', 'Random Word 2', 'Random Word 3'])

@app.route('/get_data')
def get_data():
    row_index = request.args.get('row', default=2, type=int)
    #print(f"Request received for row index: {row_index}")  # Debugging line
    if 0 <= row_index < len(ua_df):
        data = ua_df.loc[row_index].to_dict()
        #print(f"Returning data: {data}")  # Debugging line
    else:
        data = {'Error': 'Row index out of range'}
    return jsonify(data)

@app.route('/listen_updates')
def listen_updates():
    def event_stream():
        while True:
            row_index = update_queue.get()
            if row_index["color"] == "left_window_green":
                yield f'data: {{"color": "left_window_green"}}\n\n'
            elif row_index["color"] == "right_window_green":
                yield f'data: {{"color": "right_window_green"}}\n\n'
            elif row_index["color"] == "bottom_window_green":
                yield f'data: {{"color": "bottom_window_green"}}\n\n'
            elif row_index["color"] == "top_window_green":
                yield f'data: {{"color": "top_window_green"}}\n\n'
            else:
                yield f'data: {{"color": "None"}}\n\n'
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def process_video():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process frame
        for frame in process_frame(frame):
            yield frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

#if __name__ == '__main__':
#    app.run(debug=True, port=5000, threaded=True)
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)