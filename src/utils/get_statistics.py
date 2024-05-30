# mean coords for id in each frame
# mean coords for each 
import numpy as np

def get_patterns(df):

    
    n_frames = df['Frame'].values[-1]+1
    unique_ids = df.ID.unique()
    n_unique_ids = len(unique_ids)
    # only rows that where any keypoint appears
    df_not_zero = df[df['LandmarkApears']].copy()


    # inidivual scores
    # ------------------------------------------------------------------------------------------------------------------------------#


    # mean for each id in each frame
    df_mean_1 = df_not_zero[['Frame','ID','Pose_X','Pose_Y']].copy()
    df_mean_1 = df_mean_1.groupby(['Frame','ID'])[['Pose_X','Pose_Y']]\
                            .mean().round(decimals=2).add_prefix("Mean_").add_suffix("_PerID_EachFrame")\
                            .reset_index()


    # distance between face to lower limb
    df_face = df_not_zero[df_not_zero['BodyPart']=='face'].copy()
    df_lower_limbs = df_not_zero[df_not_zero['BodyPart']=='lower_limbs'].copy()
    
    df_face = df_face.groupby(['Frame','ID'])[['Pose_X','Pose_Y']]\
                            .mean().round(decimals=2).add_prefix("Face_")\
                            .reset_index()
    df_lower_limbs = df_lower_limbs.groupby(['Frame','ID'])[['Pose_X','Pose_Y']]\
                            .mean().round(decimals=2).add_prefix("LowerLimbs_")\
                            .reset_index()
    
    df_face_lower_limbd_distance = df_face.merge(df_lower_limbs,on=['Frame','ID'])

    # Euclidean Distance 
    df_face_lower_limbd_distance['FaceToLoerLimbs_Euclidean'] = \
                    np.sqrt((df_face_lower_limbd_distance['Face_Pose_X']-df_face_lower_limbd_distance['LowerLimbs_Pose_X'])**2 + 
                            (df_face_lower_limbd_distance['Face_Pose_Y']-df_face_lower_limbd_distance['LowerLimbs_Pose_Y'])**2).round(2)
    
    # Manhattan Distance 
    df_face_lower_limbd_distance['FaceToLoerLimbs_Manhattan'] = \
                    abs(df_face_lower_limbd_distance['Face_Pose_X']-df_face_lower_limbd_distance['LowerLimbs_Pose_X']) +\
                    abs(df_face_lower_limbd_distance['Face_Pose_Y']-df_face_lower_limbd_distance['LowerLimbs_Pose_Y'])

    
    df_face_lower_limbd_distance = df_face_lower_limbd_distance[['Frame','ID','FaceToLoerLimbs_Euclidean','FaceToLoerLimbs_Manhattan']]

    # number of face for each ID in each frame
    df_face = df_not_zero[df_not_zero['BodyPart']=='face'][['Frame','ID','BodyPart']].copy()
    df_face = df_face.groupby(['Frame','ID'],as_index=False)['BodyPart'].count().rename(columns={"BodyPart":"ActualFaceCount"})
    df_face['RealFaceCount'] = 5
    df_face['FaceToFrontEachFrame'] = df_face['ActualFaceCount']/df_face['RealFaceCount'].round(2) * 100
    
    # return df_face
    # df_face_ = df_face.groupby("ID")[['RealFaceCount','ActualFaceCount']].sum().reset_index()
    # df_face_['TotalFaceToFront'] = df_face_['ActualFaceCount']/df_face_['RealFaceCount'].round(2) * 100
    # df_face_ = df_face_[['ID','TotalFaceToFront']]

    # df_face = df_face[['Frame','ID','FaceToFrontEachFrame']].merge(df_face_,on='ID')


    
    # scores
    individual_scores_df = df_mean_1.merge(df_face_lower_limbd_distance,on=['Frame','ID'])
    individual_scores_df = individual_scores_df.merge(df_face,on=['Frame','ID'])
    individual_scores_df['n_frames'] = n_frames

    individual_scores_df.drop_duplicates(inplace=True)


    # total scores each frame
    # ------------------------------------------------------------------------------------------------------------------------------#


    df_mean_1 = df_not_zero.copy()
    df_mean_1 = df_mean_1.groupby('Frame')[['Pose_X','Pose_Y']]\
                            .mean().round(decimals=2).add_prefix("Mean_").add_suffix("_EachFrame")\
                            .reset_index()
    
    # distance between face to lower limb
    df_face = df_not_zero[df_not_zero['BodyPart']=='face'].copy()
    df_lower_limbs = df_not_zero[df_not_zero['BodyPart']=='lower_limbs'].copy()
    
    df_face = df_face.groupby('Frame')[['Pose_X','Pose_Y']]\
                            .mean().round(decimals=2).add_prefix("Face_")\
                            .reset_index()
    df_lower_limbs = df_lower_limbs.groupby('Frame')[['Pose_X','Pose_Y']]\
                            .mean().round(decimals=2).add_prefix("LowerLimbs_")\
                            .reset_index()
    
    df_face_lower_limbd_distance = df_face.merge(df_lower_limbs,on='Frame')

    # Euclidean Distance 
    df_face_lower_limbd_distance['FaceToLoerLimbs_Euclidean'] = \
                    np.sqrt((df_face_lower_limbd_distance['Face_Pose_X']-df_face_lower_limbd_distance['LowerLimbs_Pose_X'])**2 + 
                            (df_face_lower_limbd_distance['Face_Pose_Y']-df_face_lower_limbd_distance['LowerLimbs_Pose_Y'])**2).round(2)
    
    # Manhattan Distance 
    df_face_lower_limbd_distance['FaceToLoerLimbs_Manhattan'] = \
                    abs(df_face_lower_limbd_distance['Face_Pose_X']-df_face_lower_limbd_distance['LowerLimbs_Pose_X']) +\
                    abs(df_face_lower_limbd_distance['Face_Pose_Y']-df_face_lower_limbd_distance['LowerLimbs_Pose_Y'])

    
    df_face_lower_limbd_distance = df_face_lower_limbd_distance[['Frame','FaceToLoerLimbs_Euclidean','FaceToLoerLimbs_Manhattan']]

    # faces
    df_face = df_not_zero[df_not_zero['BodyPart']=='face'][['Frame','BodyPart']].copy()
    df_face = df_face.groupby('Frame',as_index=False)['BodyPart'].count().rename(columns={"BodyPart":"ActualFaceCount"})
    df_face['RealFaceCount'] = 5 * n_unique_ids
    df_face['FaceToFrontEachFrame'] = df_face['ActualFaceCount']/df_face['RealFaceCount'].round(2) * 100
    
    # return df_face
    # df_face_ = df_face.groupby("Frame")[['RealFaceCount','ActualFaceCount']].sum().reset_index()
    # df_face_['TotalFaceToFront'] = df_face_['ActualFaceCount']/df_face_['RealFaceCount'].round(2) * 100
    # df_face_ = df_face_[['Frame','TotalFaceToFront']]

    df_face = df_face[['Frame','FaceToFrontEachFrame']]

    # return df_face_lower_limbd_distance
    each_frame_scores = df_mean_1.merge(df_face_lower_limbd_distance,on='Frame')
    each_frame_scores = each_frame_scores.merge(df_face,on='Frame')



    # total scores each ID
    # ------------------------------------------------------------------------------------------------------------------------------#

    df_mean_1 = df_not_zero.copy()
    df_mean_1 = df_mean_1.groupby('ID')[['Pose_X','Pose_Y']]\
                            .mean().round(decimals=2).add_prefix("Mean_").add_suffix("_EachID")\
                            .reset_index()
    df_std_1 = df_not_zero.copy()
    df_std_1 = df_std_1.groupby('ID')[['Pose_X','Pose_Y']]\
                            .std().round(decimals=4).add_prefix("STD_").add_suffix("_EachID")\
                            .reset_index()
    df_mean_1 = df_mean_1.merge(df_std_1,on='ID')
    # distance between face to lower limb
    df_face = df_not_zero[df_not_zero['BodyPart']=='face'].copy()
    df_lower_limbs = df_not_zero[df_not_zero['BodyPart']=='lower_limbs'].copy()
    
    df_face = df_face.groupby('ID')[['Pose_X','Pose_Y']]\
                            .mean().round(decimals=2).add_prefix("Face_")\
                            .reset_index()
    df_lower_limbs = df_lower_limbs.groupby('ID')[['Pose_X','Pose_Y']]\
                            .mean().round(decimals=2).add_prefix("LowerLimbs_")\
                            .reset_index()
    
    df_face_lower_limbd_distance = df_face.merge(df_lower_limbs,on='ID')

    # Euclidean Distance 
    df_face_lower_limbd_distance['FaceToLoerLimbs_Euclidean'] = \
                    np.sqrt((df_face_lower_limbd_distance['Face_Pose_X']-df_face_lower_limbd_distance['LowerLimbs_Pose_X'])**2 + 
                            (df_face_lower_limbd_distance['Face_Pose_Y']-df_face_lower_limbd_distance['LowerLimbs_Pose_Y'])**2).round(2)
    
    # Manhattan Distance 
    df_face_lower_limbd_distance['FaceToLoerLimbs_Manhattan'] = \
                    abs(df_face_lower_limbd_distance['Face_Pose_X']-df_face_lower_limbd_distance['LowerLimbs_Pose_X']) +\
                    abs(df_face_lower_limbd_distance['Face_Pose_Y']-df_face_lower_limbd_distance['LowerLimbs_Pose_Y'])

    
    df_face_lower_limbd_distance = df_face_lower_limbd_distance[['ID','FaceToLoerLimbs_Euclidean','FaceToLoerLimbs_Manhattan']]

    # faces
    df_face = df_not_zero[df_not_zero['BodyPart']=='face'][['ID','BodyPart']].copy()
    df_face = df_face.groupby('ID',as_index=False)['BodyPart'].count().rename(columns={"BodyPart":"ActualFaceCount"})
    # return df_face
    df_face['RealFaceCount'] = 5 * n_frames
    df_face['FaceToFrontEachID'] = df_face['ActualFaceCount']/df_face['RealFaceCount'].round(2) * 100


    df_face = df_face[['ID','FaceToFrontEachID']]

    # return df_face_lower_limbd_distance
    each_id_scores = df_mean_1.merge(df_face_lower_limbd_distance,on='ID')
    each_id_scores = each_id_scores.merge(df_face,on='ID')

    return individual_scores_df,each_frame_scores,each_id_scores

    # return df_not_zero.shape,df.shape