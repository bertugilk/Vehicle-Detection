import cv2

video_src = 'Video/pedestrians.avi'
video = cv2.VideoCapture(video_src)

car_cascade="Cascades/cars.xml" # Araba tespiti için.
bus_cascade="Cascades/Bus.xml"  # Otobüs tespiti için.
pedestrian_cascade="Cascades/pedestrian.xml" # Yaya tespiti için.
two_wheeler_cascade="Cascades/two_wheeler.xml" # İki tekerlekli tespiti için.

def carDetection():
    car=cv2.CascadeClassifier(car_cascade)
    while True:
        ret, img = video.read()
        if (type(img) == type(None)):
            break

        grayIMG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cars = car.detectMultiScale(grayIMG, 1.1, 1)

        for (x,y,h,w) in cars:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,"Car",(x, y + h+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
        cv2.imshow("Video",img)

        if cv2.waitKey(33) == 27:
            break

def busDetection():
    bus=cv2.CascadeClassifier(bus_cascade)
    while True:
        ret, img = video.read()
        if (type(img) == type(None)):
            break

        grayIMG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        buses = bus.detectMultiScale(grayIMG, 1.16, 1)

        for (x,y,h,w) in buses:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,"Bus",(x, y + h+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
        cv2.imshow("Video",img)

        if cv2.waitKey(33) == 27:
            break

def pedestrianDetection():
    pedestrian=cv2.CascadeClassifier(pedestrian_cascade)
    while True:
        ret, img = video.read()
        if (type(img) == type(None)):
            break

        grayIMG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pedestrians = pedestrian.detectMultiScale(grayIMG, 1.3, 2)

        for (x,y,h,w) in pedestrians:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,"Pedestrian",(x, y + h+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
        cv2.imshow("Video",img)

        if cv2.waitKey(33) == 27:
            break

def two_wheelerDetection():
    two_wheeler=cv2.CascadeClassifier(two_wheeler_cascade)
    while True:
        ret, img = video.read()
        if (type(img) == type(None)):
            break

        grayIMG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        two_wheelers = two_wheeler.detectMultiScale(grayIMG, 1.16, 1)

        for (x,y,h,w) in two_wheelers:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,"Two_wheeler",(x, y + h+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
        cv2.imshow("Video",img)

        if cv2.waitKey(33) == 27:
            break

print("\n")
print("\t1- Cars")
print("\t2- Buses")
print("\t3- Pedestrian")
print("\t4- Two_wheeler\n")

select=int(input("Which vehicle would you like to detect?\n"))

if select==1:
    carDetection()
elif select==2:
    busDetection()
elif select==3:
    pedestrianDetection()
elif select==4:
    two_wheelerDetection()

cv2.destroyAllWindows()