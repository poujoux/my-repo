from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os


class EducateIt:

    modulepath = os.path.join("/root", "prjct", "model.pkl")

    def __init__(self, image):
        self.image = image


    def putfiles(self, name):
        info = []
        plantforinfo = []

        data = self.imginfo()

        info.append(data)
        plantforinfo.append(name)

        # Train after all self.images have been processed
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(info, plantforinfo)

        joblib.dump(model, EducateIt.modulepath)

        return 0

    def predict(self):
        if not os.path.exists(EducateIt.modulepath):
            return -1
        
        model = joblib.load(EducateIt.modulepath)

        data = self.imginfo()

        prediction = model.predict([data])

        return prediction[0]

    def imginfo(self):
        data = []

        img = Image.open(self.image)
        img = img.convert("RGB")
        img = img.resize((40, 40))

        for pixel in img.getdata():
            data.extend(pixel)

        return data

