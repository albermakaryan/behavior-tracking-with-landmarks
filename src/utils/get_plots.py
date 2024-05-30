
import matplotlib.pyplot as plt
import numpy as np

def plot_individual_results(df,suptitle=None):

    df.sort_values(['Frame','ID'],inplace=True)


    # unique ids
    unique_ids = df.ID.unique()
    n_unique = len(unique_ids)
    # n_frames = df['n_frames'].values[0]
    # x_axis = np.arange(1,n_frames+1)

    # fig,ax = plt.subplots(n_unique,3)

    # title
    # suptitle = "Visualization of metrics for behavior for each ID in each frame" if suptitle is None else suptitle
    # fig.suptitle(suptitle,fontsize=20)
    
    
    # fig.set_figwidth(15)
    # fig.set_figheight(15)
    # fig.subplots_adjust(left=0.1, bottom=.10, right=0.1, top=0.9, wspace=0.5, hspace=0.8)    
    # fig.subplots_adjust(hspace=0.8)
    
    for ind in range(n_unique):

        ID = unique_ids[ind]


        fig,ax = plt.subplots(1,3)
        fig.set_figwidth(20)
        fig.set_figheight(5)
        fig.suptitle("Visualization of metrics for behavior for ID [" + str(ID)+"] in each frame",fontsize=20)


        
        id_df = df[df['ID'] == ID].copy()

        # mean coordinates

        mean_pose_x = id_df['Mean_Pose_X_PerID_EachFrame'].values
        mean_pose_y = id_df['Mean_Pose_Y_PerID_EachFrame'].values
        face_feet_distance_ec = id_df['FaceToLoerLimbs_Euclidean'].values
        face_feet_distance_mn = id_df['FaceToLoerLimbs_Manhattan'].values
        face_to_front = id_df['FaceToFrontEachFrame'].values
        
        x_axis = id_df['Frame'].values

        # mean points
        ax[0].plot(x_axis,mean_pose_x,label='Mean keypoints X')
        ax[0].plot(x_axis,mean_pose_y,label='Mean keypoints Y')
        ax[0].set_xlabel("Frame")
        ax[0].set_ylabel("Coordinate value")
        ax[0].legend()
        ax[0].set_title("Mean of keypoints for each frame")

        # distances
        ax[1].plot(x_axis,face_feet_distance_ec,label='Euclidean')
        ax[1].plot(x_axis,face_feet_distance_mn,label='Manhattan')
        ax[1].set_xlabel("Frame")
        ax[1].set_ylabel("Distance")
        ax[1].legend()
        ax[1].set_title("Distance between face to lower limbs")


        # face to front
        ax[2].plot(x_axis,face_to_front)
        ax[2].set_ylabel("Precent")
        # i][2].legend()
        ax[2].set_xlabel("Frame")

        ax[2].set_title("Face to front ratio")
        # break
        
    plt.show()
    # break






def plot_total_metrics_each_frame(df,suptitle=None):


    fig,ax = plt.subplots(3,1)

    # title
    suptitle = "Visualization of metrics for behavior in each frame" if suptitle is None else suptitle
    fig.suptitle(suptitle,fontsize=20)
    
    
    fig.set_figwidth(15)
    fig.set_figheight(10)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)

    x_axis = df['Frame'].values
    pose_x = df['Mean_Pose_X_EachFrame'].values
    pose_y = df['Mean_Pose_Y_EachFrame'].values
    ec = df['FaceToLoerLimbs_Euclidean'].values
    mn = df['FaceToLoerLimbs_Manhattan'].values
    ftf = df['FaceToFrontEachFrame'].values

    ax[0].plot(x_axis,pose_x,label='Keypoints X')
    ax[0].plot(x_axis,pose_y,label='Keyponts Y')
    ax[0].set_xlabel("Frame")
    ax[0].set_ylabel("Coordinate value")
    ax[0].set_title("Mean coordinates of keypoints for each frame")
    ax[0].legend()

    ax[1].plot(x_axis,ec,label='Euclidean')
    ax[1].plot(x_axis,mn,label='Manhattan')
    ax[1].set_xlabel("Frame")
    ax[1].set_ylabel("Distance")
    ax[1].set_title("Mean face to lower limbs distance for each frame") 
    ax[1].legend()

    ax[2].plot(x_axis,ftf)
    ax[2].set_xlabel("Frame")
    ax[2].set_ylabel("Percent")
    ax[2].set_title("Face to front ratio") 
    plt.show()
    
    
    
    
def plot_total_metrics_each_id(data,suptitle=None,break_soon=False):
    

    df = data[data['LandmarkApears']].copy()

    unique_ids = df.ID.unique()
    n_unique_ids = len(unique_ids)

    key_points = df.BodyPart.unique()
    n_keypoints = len(key_points)

    for i in range(n_unique_ids):

        ID = unique_ids[i]

        id_df = df[df['ID'] == ID]

        fig,ax = plt.subplots(n_keypoints,2)

        fig.suptitle("Distribution of body parts coordinate movements for ID: " + str(ID),fontsize=20)

        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
        fig.set_figheight(15)
        fig.set_figwidth(15)

        for i_key in range(n_keypoints):

            keypoint = key_points[i_key]

            id_df_key = id_df[id_df['BodyPart'] == keypoint]

            pose_x,pose_y = id_df_key['Pose_X'],id_df_key['Pose_Y']
            x_axis = id_df_key['Frame'].values

            if keypoint == "lower_limbs":
                title = "Feet"
            elif keypoint == "upper_limbs":
                title = "Hand"
            else:
                title = keypoint
                
            ax[i_key][0].hist(pose_x)
            ax[i_key][0].set_xlabel("X coordinates")
            ax[i_key][0].set_title(title.capitalize())
            
            ax[i_key][1].hist(pose_y)
            ax[i_key][1].set_xlabel("Y coordinates")
            ax[i_key][1].set_title(title.capitalize())
        plt.show()
        
        if break_soon:
            break
        
    pass
    


        
        # pass

    # return unique_ids