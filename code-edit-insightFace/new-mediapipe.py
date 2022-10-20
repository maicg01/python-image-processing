import numpy as np
import mediapipe as mp
import cv2
import math

def x_element(elem):
    return elem[0]
def y_element(elem):
    return elem[1]

cap = cv2.VideoCapture(0)
pTime = 0
faceXY = []
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=5, min_detection_confidence=.9, min_tracking_confidence=.01)
drawSpec = mpDraw.DrawingSpec(0,1,1)

success, img = cap.read()
height, width = img.shape[:2]
# size = img.shape

def face_orientation(frame, faceXY):
    size = frame.shape #(height, width, color_channel)

    image_points = np.array([
        faceXY[1],      # "nose"
        faceXY[152],    # "chin"
        faceXY[226],    # "left eye"
        faceXY[446],    # "right eye"
        faceXY[57],     # "left mouth"
        faceXY[287]     # "right mouth"
    ], dtype="double")

    

                        
    # model_points = np.array([
    #                         (0.0, 0.0, 0.0),             # Nose tip
    #                         (0.0, -330.0, -65.0),        # Chin
    #                         (-165.0, 170.0, -135.0),     # Left eye left corner
    #                         (165.0, 170.0, -135.0),      # Right eye right corne
    #                         (-150.0, -150.0, -125.0),    # Left Mouth corner
    #                         (150.0, -150.0, -125.0)      # Right mouth corner                         
    #                     ])


    model_points = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left Mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
    ], dtype=np.float64)

    # Camera internals
 
    center = (size[1]/2, size[0]/2)
    focal_length = size[1]
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    
    # axis = np.float32([[500,0,0], 
    #                       [0,500,0], 
    #                       [0,0,500]])

    axis = np.array([(0.0, 0.0, 1000.0)])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), image_points, faceXY[1]


t = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:                                            # if faces found
        dist=[]
        for faceNum, faceLms in enumerate(results.multi_face_landmarks):                            # loop through all matches
            # mpDraw.draw_landmarks(img, faceLms, landmark_drawing_spec=drawSpec) # draw every match
            faceXY = []
            for id,lm in enumerate(faceLms.landmark):                           # loop over all land marks of one face
                ih, iw, _ = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                # print(lm)
                faceXY.append((x, y))                                           # put all xy points in neat array

            imgpts, modelpts, rotate_degree, image_points, nose = face_orientation(imgRGB, faceXY)

            for i in image_points:
                cv2.circle(img,(int(i[0]),int(i[1])),4,(255,0,0),-1)
            maxXY = max(faceXY, key=x_element)[0], max(faceXY, key=y_element)[1]
            minXY = min(faceXY, key=x_element)[0], min(faceXY, key=y_element)[1]

            xcenter = (maxXY[0] + minXY[0]) / 2
            ycenter = (maxXY[1] + minXY[1]) / 2

            dist.append((faceNum, (int(((xcenter-width/2)**2+(ycenter-height/2)**2)**.4)), maxXY, minXY))     # faceID, distance, maxXY, minXY

            print(image_points)

            # (success, rotation_vector, translation_vector) = cv2.solvePnP(face3Dmodel, image_points,  camera_matrix, dist_coeffs)
            # (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))

            cv2.line(img, p1, p2, (255, 0, 0), 2)

        dist.sort(key=y_element)
        # print(dist)

        for i,faceLms in enumerate(results.multi_face_landmarks):
            if i == 0:
                cv2.rectangle(img,(dist[i][2][0],dist[i][2][1]) ,dist[i][3],(0,255,0),2)
            else:
                cv2.rectangle(img, dist[i][2], dist[i][3], (0, 0, 255), 2)
        lt = ['roll', 'pitch', 'yaw']
        k=0
        for j in range(len(rotate_degree)):
            cv2.putText(img, (lt[j] + '  ''{:05.2f}' ).format(float(rotate_degree[j])), (10, 30 + (50 * j)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
            if -5 <= float(rotate_degree[j]) <= 5:
                k += 1

            if k == 3:
                print('true')
                cv2.imwrite('./outputs/frame%s.jpg'%str(t), img)  

    t +=1

    cv2.imshow("Image", img)
    cv2.waitKey(1)