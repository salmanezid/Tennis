import random

from flask import Flask,render_template,url_for,request,redirect,session,send_file
from datetime import datetime
import random
from io import BytesIO
import os
from moviepy.editor import VideoFileClip
from utils import (read_video,
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
import constants
from trackers import PlayerTracker,BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
from copy import deepcopy
from ultralytics import YOLO
import numpy as np

def detect_players(frame, model, target_class=0, exclude_classes=[1, 2]):
    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()
    player_positions = []
    for det in detections:
        if len(det) == 6:
            x1, y1, x2, y2, conf, cls = map(int, det)
            if cls == target_class:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                player_positions.append((center_x, center_y))
            elif cls not in exclude_classes:
                continue
        elif len(det) == 4:
            x1, y1, x2, y2 = map(int, det)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            player_positions.append((center_x, center_y))
    return player_positions

def create_simplified_tennis_court_background(frame_width, frame_height):
    court = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    court[:] = (0, 128, 0)
    cv2.line(court, (0, frame_height // 2), (frame_width, frame_height // 2), (255, 255, 255), 5)
    return court

def create_player_distribution_map(video_path, model, output_image_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    movement_map = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    frame_count = 0
    previous_positions_player_1 = []
    previous_positions_player_2 = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        player_positions = detect_players(frame, model)
        for i, (center_x, center_y) in enumerate(player_positions):
            if 0 <= center_x < frame_width and 0 <= center_y < frame_height:
                if i == 0:
                    cv2.circle(movement_map, (center_x, center_y), 5, (0, 0, 255), -1)
                    if len(previous_positions_player_1) > 1:
                        for j in range(1, len(previous_positions_player_1)):
                            cv2.line(movement_map, previous_positions_player_1[j-1], previous_positions_player_1[j], (0, 0, 255), 1)
                    previous_positions_player_1.append((center_x, center_y))
                elif i == 1:
                    cv2.circle(movement_map, (center_x, center_y), 5, (255, 255, 255), -1)
                    if len(previous_positions_player_2) > 1:
                        for j in range(1, len(previous_positions_player_2)):
                            cv2.line(movement_map, previous_positions_player_2[j-1], previous_positions_player_2[j], (255, 255, 255), 1)
                    previous_positions_player_2.append((center_x, center_y))
        frame_count += 1
    cap.release()
    court = create_simplified_tennis_court_background(frame_width, frame_height)
    result_image = cv2.addWeighted(court, 1, movement_map, 0.7, 0)
    blurred_map = cv2.GaussianBlur(result_image, (15, 15), 0)
    contrast_map = cv2.convertScaleAbs(blurred_map, alpha=2.5, beta=0)
    cv2.imwrite(output_image_path, contrast_map)
    print(f"Player distribution map saved at {output_image_path}")
    print(f"Processed {frame_count} frames.")


def main(input_video_path,output_vedio_path ,output_image_path):
    # Read Video
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/last.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path=None
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path=None
                                                     )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)


    # Court Line Detector model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # MiniCourt
    mini_court = MiniCourt(video_frames[0])

    # Detect ball shots
    ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)
    print("chofni",player_detections)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                          ball_detections,
                                                                                                          court_keypoints)

    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]

    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           )

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # player who the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                 ball_mini_court_detections[start_frame][1]))

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           )

        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']



    # Draw output
    ## Draw Player Bounding Boxes
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames= ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw court Keypoints
    output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))

    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, output_vedio_path)

    #----Gnerate Image de Nuage----#
    try:
      model = YOLO("yolov8n.pt")
      print("Model loaded successfully!")
    except Exception as e:
      print(f"Error loading model: {e}")

    output_image_path = "outputs/nuageplayers.jpg"
    create_player_distribution_map(input_video_path, model, output_image_path)

def convert_avi_to_mp4(input_path, output_path):
    """
    Converts a .avi video file to .mp4 format.

    Args:
        input_path (str): Path to the input .avi file.
        output_path (str): Path to save the converted .mp4 file.

    Returns:
        None
    """
    try:
        # Load the .avi file
        video = VideoFileClip(input_path)

        # Write the video to an .mp4 file
        video.write_videofile(output_path, codec='libx264', audio_codec='aac')

        print(f"Conversion successful! File saved at: {output_path}")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
app = Flask(__name__)
UPLOAD_FOLDER = 'static/videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/',methods=['POST','GET'])
def dashboard():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400

        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        try:
            # Upload the file directly from memory to S3
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)
            random_i = str(random.randint(1, 10000))
            main(video_path, f"static/videos/{random_i}.avi", f"static/CSS/images/{random_i}.jpg")
            convert_avi_to_mp4(f"static/videos/{random_i}.avi", f"static/videos/{random_i}.mp4")

            video_url = url_for('static', filename=f'videos/{random_i}.mp4')
            image_url = url_for('static', filename=f'CSS/images/{random_i}.jpg')
            return render_template('output.html',video_url=video_url,image=image_url)
        except Exception as e:
            return f"An error occurred: {e}", 500

    return render_template('upload.html')
@app.route('/output',methods=['POST','GET'])
def output():

    return render_template('output.html')


if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0')