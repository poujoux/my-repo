from modules.predict import EducateIt 
from flask import Flask, request, render_template, jsonify
from modules.apisendai import sendai # relative imports from same package
from dotenv import load_dotenv

import os

app = Flask(__name__, template_folder="templates")

load_dotenv()

api_key = os.getenv("api_key")
apikeym = os.getenv("mastodon_api_key")

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/educateit", methods=["POST"])
def educate():
    imgs = request.files.get("efiles")
    print(12*"-", "imgs:", imgs)

    if not imgs:
        return jsonify({"No images uploaded"}), 400

    
    obj = EducateIt(imgs)
    print(12*"-", "The object has been created")

    #prediction = obj.predict()
    prediction = -1
    print(12*"-", "Prediction", prediction)

    if prediction == -1:

        predictionobj = sendai(api_key)
        predictiontxt = predictionobj.imggguess(imgs)

        print(12*"-", "Prediction...", predictiontxt)
        return jsonify({"msg": predictiontxt}), 400


    result = obj.putfiles(predictiontxt)
    print(12*"-", "That guess")

    return jsonify({"msg": prediction}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    
