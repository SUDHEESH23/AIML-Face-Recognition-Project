import cv2 as cv
import argparse
import numpy as np


def visualize(image,face,thickness = 2):
  for idx, face in enumerate(face[1]):
    coords = face[:-1].astype(np.int32) 
    cv.rectangle(image,(coords[0],coords[1]),(coords[0]+coords[2], coords[1]+coords[3]),(0,255,0), thickness )

    cv.circle(image,(coords[4], coords[5]),2,(255,0,0),thickness)
    cv.circle(image,(coords[6], coords[7]),2,(0,0,255),thickness)
    cv.circle(image,(coords[8], coords[9]),2,(0,255,0),thickness)
    cv.circle(image,(coords[10], coords[11]),2,(255,0,255),thickness)
    cv.circle(image,(coords[12], coords[13]),2,(0,255,255),thickness)

ap=argparse.ArgumentParser()
ap.add_argument("-r", "--reference_image", required=True, help="reference.jpg")
ap.add_argument("-q", "--query_image", required=True, help="query.jpg")
args=vars(ap.parse_args())
ref_image= cv.imread(args["reference_image"]) #read the image & send the ref_image
query_image= cv.imread(args["query_image"])

score_threshold = 0.9
nms_threshold = 0.3
top_k = 5000                                                                
faceDetector= cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx","",(ref_image.shape[1],ref_image.shape[0]),score_threshold,nms_threshold, top_k)
faceInAdhaar = faceDetector.detect(ref_image)
visualize(ref_image,faceInAdhaar)

cv.imshow("face",ref_image) # the image we send
cv.waitKey(0)

faceDetector.setInputSize((query_image.shape[1] , query_image.shape[0]))
faceInQuery= faceDetector.detect(query_image)
visualize(query_image,faceInQuery)

cv.imshow("face", query_image)
cv.waitKey(0)
