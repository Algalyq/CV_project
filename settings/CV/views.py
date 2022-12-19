from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *
from .utils import *

from .Backend import *

def index(request):
    return render(request, 'CV/index.html')

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
        camera = cv.VideoCapture(0)
        face_classifier = cv.CascadeClassifier('Classifiers/haarface.xml')


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
            cv.imshow('Face', img)

        camera.release()
        cv.destroyAllWindows()
        create_train()
        return redirect('index')
  
    return render(request, 'CV/takeCard.html')


def recognition(request):
    id_names = pd.read_csv('id-names.csv')
    id_names = id_names[['id', 'name']]
    faceClassifier = cv.CascadeClassifier('Classifiers/haarface.xml')
    lbph = cv.face.LBPHFaceRecognizer_create(threshold=500)
    lbph.read('Classifiers/TrainedLBPH.yml')
    font = cv.FONT_HERSHEY_SIMPLEX
    camera = cv.VideoCapture(0)

    while cv.waitKey(1) & 0xFF != ord('q'):
        _, img = camera.read()
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        label = 0
        userId = 0
        faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)
        name = ''
        for x, y, w, h in faces:
            faceRegion = grey[y:y + h, x:x + w]
            faceRegion = cv.resize(faceRegion, (220, 220))

            label, trust = lbph.predict(faceRegion)
            trust = 100 - int(trust)
            names = id_names[id_names['id'] == label]['name'].item()
            name = names
            if trust > 55:
                userId = label
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255,0), 2)  
                cv.putText(img, "Detected" , (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255,0))
            else:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0,255), 2) 
                cv.putText(img, "Unknown",  (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255,0))
        cv.imshow("Face",img)
        if(cv.waitKey(1) == ord('q')):
            break
        elif(userId != 0):
            cv.waitKey(1000)
            camera.release()
            cv.destroyAllWindows()
            return redirect('profile',name)

    camera.release()
    cv.destroyAllWindows()
    return redirect('/')        

       


def profile(request,data):
    return render(request,'CV/index.html',{'name':data})


 