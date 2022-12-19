import os
import pandas as pd
import numpy as np
import cv2 as cv

id_names = pd.read_csv('/home/nurzhan/Desktop/CV_project/settings/id-names.csv')
id_names = id_names[['id', 'name']]

lbph = cv.face.LBPHFaceRecognizer_create(threshold=500)


def create_train():
    faces = []
    labels = []
    for id in os.listdir('/home/nurzhan/Desktop/CV_project/settings/faces'):
        path = os.path.join('/home/nurzhan/Desktop/CV_project/settings/faces', id)
        try:
            os.listdir(path)
        except:
            continue
        for img in os.listdir(path):
            try:
                face = cv.imread(os.path.join(path, img))
                face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
                faces.append(face)
                labels.append(int(id))
            except:
                pass
    return np.array(faces), np.array(labels)
faces, labels = create_train()
print('Training Started')
lbph.train(faces, labels)
lbph.save('/home/nurzhan/Desktop/CV_project/settings/Classifiers/TrainedLBPH.yml')
print('Training Complete!')
