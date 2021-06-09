import cv2,time
import pandas
from datetime import datetime

#this will store the very first frame that will be captured
first_frame=None
#stores 1 in list if object was present ;0 if object was not detected
status_list=[None,None]
#stores the time of arrival and departure of detected objects
times=[]
#stores all arrival and exit time of detected objects
df=pandas.DataFrame(columns=["start","end"])

#stat cpaturing video from webcam
video = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    #video.read() checks whether video is being captured and the cpatured frame is stored in frame
    check,frame=video.read()
    #as intially none object is detected
    status=0
    #turning image gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #making gray image gaussian blur
    gray=cv2.GaussianBlur(gray,(21,21),0)
    #saving the first captured image in first_frame so tht with respect to it we can identify arrival of some object in front of webcam
    if first_frame is None:
        first_frame=gray
        continue
    #this provides a 3d matrix which stores the difference in intensity of pixels in the two images:first_frame and gray;places where the intensity is 0 means in both the images the intensity(color) of that pixel was same that is why on taking difff result was 0 and therefore those parts are common to both images;places where we get significant intensity(color)changes mark the arrival of some new object as compared to the first_frame
    delta_frame=cv2.absdiff(first_frame,gray)
    #threshold is used to differentiate object from background by converting the image matrix to binary;thresholding will be done on delta_frame;threshold value is 30 that means the pixels where intensity diff is 30 will be marked as black means will be included as background whereas the pixels or points on image having intensity greater than 30 will be given 255 color white which indicates the object in image.Thresholding is done by THRESH_BINARY METHOD  that is for intensity diff less than 30 assign black color to those pixels and above 30 white or the specified color(here 255 means white) 
    thresh_delta=cv2.threshold(delta_frame,100,255,cv2.THRESH_BINARY)[1]
    #this increases white part
    thresh_delta = cv2.dilate(thresh_delta, None, iterations = 0)
    #to find boundaries around objects detected in image,we use findContours;it detects the points with same intensity to build up boundary(here with white);copy of binary black white thresholded image is passed,2nd arg is mode of finding contour that is in retr_tree mode all contour points are retrieved ;3rd arg tells the method of selecting points-chain_simple method picks only the corner pnts of the boundary whereas chain_none picks all points of the boundary of detected object 
    cnts,_=cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #goes throgh all the boundaries to detected object;if area bounded is less than 1000,the object is ignored while the object greater than 1000 are detected and therefore status which marks detection of objects is changed to 1
    
    for contour in cnts:
        if cv2.contourArea(contour)<10000:
            continue
        status=1
        #boundingreact returns the x,y widht and height of the detected boundary
        (x,y,w,h)=cv2.boundingRect(contour)
        #rectangle is placed around detected object
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
    # status of each frame is appended in this list;status is 1 if object is detected in frame wrt to first_frame;marked as 0 if no object detected wrt to    
    status_list.append(status)
    #we compare between present frame and the just previous frame for detection
    status_list=status_list[-2:]
    #if now the object is detected (status is 1) and beofre this the there was no object status 0,then this time is marked as arrival point of object
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now().time())
    #previous frame had object and present frame doesnt;this marks exit of object
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now().time())
        
    cv2.imshow("frame",frame)
    
    cv2.imshow("thresh",thresh_delta)
    key=cv2.waitKey(1)
    #if q is present video capturing stops
    if key==ord("q"):
        break
video.release()
cv2.destroyAllWindows()
print(status_list)
print(times)
#tabulating the start and end time of object detection
for i in range(0,len(times)-1,2):
    df=df.append({"start":times[i],"end":times[i+1]},ignore_index=True)
print(df)
df.to_csv("motion.csv")
    
    
    