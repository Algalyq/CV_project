


def taken():

    face_classifier = cv.CascadeClassifier('Classifiers/haarface.xml')
    img = cv.imread('')
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        face_region = grey[y:y + h, x:x + w]
        if  np.average(face_region) > 50:
            face_img = cv.resize(face_region, (220, 220))
            img_name = f'face.{id}.{datetime.now().microsecond}.jpeg'
            cv.imwrite(f'faces/{id}/{img_name}', face_img)
            print(f'{photos_taken} -> Photos taken!')
