
import cv2
import numpy as np
import pandas as pd
import os


def get_yolov8_pose_track_coords(pose_result,save_dir,file_name):


    landmarks_list = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", 
             "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", 
             "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

    landmark_body_part = {'nose': "face",
                     'left_eye': "face",
                     'right_eye': "face",
                     'left_ear': "face",
                     'right_ear': "face",
                     'left_shoulder':"upper_limbs",
                     'right_shoulder': "upper_limbs",
                     'left_elbow': "upper_limbs",
                     'right_elbow': "upper_limbs",
                     'left_wrist': "upper_limbs",
                     'right_wrist': "upper_limbs",
                     'left_hip': "lower_limbs",
                     'right_hip': "lower_limbs",
                     'left_knee': "lower_limbs",
                     'right_knee': "lower_limbs",
                     'left_ankle': "lower_limbs",
                     'right_ankle': "lower_limbs"}
    
    key_landmarks = {'nose': "nose",
                     'left_eye': "eye",
                     'right_eye': "eye",
                     'left_ear': "ear",
                     'right_ear': "ear",
                     'left_shoulder':"left_hand",
                     'right_shoulder': "right_hand",
                     'left_elbow': "left_hand",
                     'right_elbow': "right_hand",
                     'left_wrist': "left_hand",
                     'right_wrist': "right_hand",
                     'left_hip': "left_foot",
                     'right_hip': "right_foot",
                     'left_knee': "left_foot",
                     'right_knee': "right_foot",
                     'left_ankle': "left_foot",
                     'right_ankle': "right_foot"}
    
    key_landmarks_as_body_part = {
                                'nose': "nose",
                                'left_eye': "eye",
                                'right_eye': "eye",
                                'left_ear': "ear",
                                'right_ear': "ear",
                                'left_shoulder':"hand",
                                'right_shoulder': "hand",
                                'left_elbow': "hand",
                                'right_elbow': "hand",
                                'left_wrist': "hand",
                                'right_wrist': "hand",
                                'left_hip': "foot",
                                'right_hip': "foot",
                                'left_knee': "foot",
                                'right_knee': "foot",
                                'left_ankle': "foot",
                                'right_ankle': "foot"}

    n_landmarks = len(landmarks_list)
    # cap = cv2.VideoCapture(video_path)

    n_frames = len(pose_result)

    # print(n_frames)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 0, 0)  
    thickness = 2
    
    
    full_cord_df = None
    
    
    dict_for_keypoints = {}
    
    for i_frame in range(n_frames):
    
    
        print(f"{'-'*30} FRAME {i_frame+1}/{n_frames} {'-'*30}")
        # plt.figure(figsize=(15,150))
    
    
        # get the frame
        result = pose_result[i_frame]
    
        orig_image = result.orig_img.copy() # get original image
        boxes = result.boxes # get bounding boxes
        keypoints = result.keypoints
    
        n_objects = len(result)
        
        for i_obj in range(n_objects):
    
            box = boxes[i_obj]
            keypoint = keypoints[i_obj]
    
            cls = int(box.cls.item())
            ID = int(box.id.item())
    
            box_x_min,box_y_min,box_x_max,box_y_max = box.xyxy.cpu().numpy()[0].astype(int)
    
            orig_image = cv2.rectangle(orig_image,(box_x_min,box_y_min),(box_x_max,box_y_max),(0,255,0),2)
            orig_image = cv2.putText(orig_image,"ID: "+str(ID),(box_x_min,box_y_min-5),font,font_scale,font_color,thickness)
    
            keypoint_coords = keypoint.xy.cpu().numpy()[0]
    
            # store keypoints in a dictionary to analyse movements
            # if id is not in the dict, add
    
            # id_dict_key = list(dict_for_keypoints.keys())
    
            # if ID not in id_dict_key:
    
                # print("Not")
                # print(landmarks_dict)
                # dict_for_keypoints[ID] = {landmarks_list[i]:[] for i in range(n_landmarks)}.copy()
                # dict_for_keypoints[ID] = copy.deepcopy(landmarks_dict)
            
    
            for pose_i in range(n_landmarks):
    
                pose = keypoint_coords[pose_i]
    
                landmark = landmarks_list[pose_i]
    
                pose_x,pose_y = pose.astype(int)
    
                # if landmarks appears or not
                lanmark_appears = False if  np.all(pose==np.array([0,0])) else True
    
                # dict_for_keypoints[ID][landmark].append([pose_x,pose_y])
    
    
                orig_image = cv2.circle(orig_image,(pose_x,pose_y),5,(0,0,255),-1)
    
                body_part = landmark_body_part[landmark]
                
                
                key_landmark = key_landmarks[landmark]
                key_landmark_as_body_part = key_landmarks_as_body_part[landmark]
    
                inner_df = pd.DataFrame({"Frame":[i_frame],
                                         "ID":[ID],
                                         "Landmark":[landmark],
                                         "LandmarkApears": [lanmark_appears],
                                         "BodyPart":[body_part],
                                         "RightLeftKeyLandmark":[key_landmark],
                                         "KeyLandmark" : [key_landmark_as_body_part],
                                         "Pose_X":[pose_x],
                                         "Pose_Y":[pose_y],
                                         "bbox_x_min":[box_x_min],
                                         "bbox_y_min":[box_y_min],
                                         "bbox_x_max":[box_x_max],
                                         "bbox_y_max":[box_y_max]
                                        })
    
                if full_cord_df is None:
                    full_cord_df = inner_df.copy()
    
                    continue
    
                full_cord_df = pd.concat([full_cord_df,inner_df])


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # full_cord_df.to_csv(os.path.join(save_dir,file_name),index=False)
        
    full_cord_df.to_csv(os.path.join(save_dir,file_name),index=False)
    
    return full_cord_df
        