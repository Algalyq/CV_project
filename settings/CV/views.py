from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *
from .utils import *

from .Backend import *

def index(request):
    return render(request, 'CV/takeCard.html')

def takeImg(request):
    if request.method == "POST":
        if os.path.exists('id-names.csv'):
            id_names = pd.read_csv('id-names.csv')
            id_names = id_names[['id', 'name']]
        else:
            id_names = pd.DataFrame(columns=['id', 'name'])
            id_names.to_csv('id-names.csv')

        if not os.path.exists('faces'):
            os.makedirs('faces')

        # print('Welcome!')
        # print('\nPlease put in your ID.')
        # print('If this is your first time choose a random ID between 1-10000')

        id = request.POST['id']
        name = ''

        if id in id_names['id'].values:
            name = id_names[id_names['id'] == id]['name'].item()
            print(f'Welcome Back {name}!!')
        else:
            name = request.POST['name']
            os.makedirs(f'faces/{id}')
            id_names = id_names.append({'id': id, 'name': name}, ignore_index=True)
            id_names.to_csv('id-names.csv')

        # print("\nLet's capture!")

    
        # print("Now this is where you begin taking photos. Once you see a rectangle around your face, press the 's' key to capture a picture.", end=" ")
        # print("It is recommended to take atleast 20-25 pictures, from different angles, in different poses, with and without specs, you get the gist.")
        # input("\nPress ENTER to start when you're ready, and press the 'q' key to quit when you're done!")

        camera = cv.VideoCapture(0)
        face_classifier = cv.CascadeClassifier('Classifiers/haarface.xml')

        photos_taken = 0

        while(cv.waitKey(1) & 0xFF != ord('q')):
            _, img = camera.read()
            grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            for (x, y, w, h) in faces:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                face_region = grey[y:y + h, x:x + w]
                if cv.waitKey(1) & 0xFF == ord('s') and np.average(face_region) > 50:
                    face_img = cv.resize(face_region, (220, 220))
                    img_name = f'face.{id}.{datetime.now().microsecond}.jpeg'
                    cv.imwrite(f'faces/{id}/{img_name}', face_img)
                    photos_taken += 1
                    print(f'{photos_taken} -> Photos taken!')

            cv.imshow('Face', img)

        camera.release()
        cv.destroyAllWindows()
        create_train()
        return redirect('index')

    
  
    return render(request, 'CV/takeCard.html')


def Recog(request):
    id_names = pd.read_csv('id-names.csv')
    id_names = id_names[['id', 'name']]

    faceClassifier = cv.CascadeClassifier('Classifiers/haarface.xml')

    lbph = cv.face.LBPHFaceRecognizer_create(threshold=500)
    lbph.read('Classifiers/TrainedLBPH.yml')
    font = cv.FONT_HERSHEY_SIMPLEX
    camera = cv.VideoCapture(0)

    # while True:
    #     _, img = camera.read()

    #     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     faces = faceClassifier.detectMultiScale(gray, 1.3, 5)
    #     for(x,y,w,h) in faces:
    #         cv.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
    #         getId,conf = lbph.predict(gray[y:y+h, x:x+w]) #This will predict the id of the face

    #         #print conf;
    #         if conf<35:
    #             userId = getId
    #             cv.putText(img, "Detected",(x,y+h), font, 2, (0,255,0),2)
    #         else:
    #             cv.putText(img, "Unknown",(x,y+h), font, 2, (0,0,255),2)
    #     cv.imshow("Face",img)
    #     if(cv.waitKey(1) == ord('q')):
    #         break
    #     elif(userId != 0):
    #         cv.waitKey(1000)
    #         cam.release()
    #         cv2.destroyAllWindows()
    #         return redirect('/records/details/'+str(userId))

    # cam.release()
    # cv2.destroyAllWindows()
    # return redirect('/')
        # _, frame = camera.read()
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # faces = faceClassifier.detectMultiScale(gray,1.1,4)

        # for (x,y,w,h) in faces:
        #     roi_gray = gray[y:y+h,x:x+w]
        #     id,confidence = lbph.predict(roi_gray)
        #     confidence = 100 - int(confidence)
        #     pred = 0
        #     if confidence > 60:
                    #if u want to print confidence level
                            #confidence = 100 - int(confidence)
        #                 pred += +1
        #                 name = id_names[id_names['id'] == id]['name'].item()
        #                 text = name.upper()
        #                 font = cv.FONT_HERSHEY_PLAIN
        #                 frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #                 frame = cv.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv.LINE_AA)

        #     else:   
        #                 pred += -1
        #                 text = "UnknownFace"
        #                 font = cv.FONT_HERSHEY_PLAIN
        #                 frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #                 frame = cv.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv.LINE_AA)

        # cv.imshow("image", frame)


        # if cv.waitKey(20) & 0xFF == ord('q'):
        #     print(pred)
        #     if pred > 0 : 
        #         dim =(124,124)
        #         img = cv.imread(f".\\data\\{name}\\{pred}{name}.jpg", cv.IMREAD_UNCHANGED)
        #         resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        #         cv.imwrite(f".\\data\\{name}\\50{name}.jpg", resized)
        #         Image1 = Image.open(f".\\2.png") 
                      
        #             # make a copy the image so that the  
        #             # original image does not get affected 
        #         Image1copy = Image1.copy() 
        #         Image2 = Image.open(f".\\data\\{name}\\50{name}.jpg") 
        #         Image2copy = Image2.copy() 
                      
        #             # paste image giving dimensions 
        #         Image1copy.paste(Image2copy, (195, 114)) 
                      
        #             # save the image  
        #         Image1copy.save("end.png") 
        #         frame = cv.imread("end.png", 1)

        #         cv.imshow("Result",frame)
        #         cv.waitKey(5000)
        #     break

    
    # camera.release()
    # cv.destroyAllWindows()
    # return redirect('index')


    while cv.waitKey(1) & 0xFF != ord('q'):
        _, img = camera.read()
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)

        for x, y, w, h in faces:
            faceRegion = grey[y:y + h, x:x + w]
            faceRegion = cv.resize(faceRegion, (220, 220))

            label, trust = lbph.predict(faceRegion)
            try:
                if trust > 50:
                    
                    name = id_names[id_names['id'] == label]['name'].item()
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255,0), 2)
                    cv.putText(img, name, (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255,0))
            except:
                pass

        cv.imshow('Recognize', img)

    camera.release()
    cv.destroyAllWindows()
    return redirect('index')


def tema(request):
    return render(request,'CV/tema.html')


 