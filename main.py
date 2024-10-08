# 1. Library imports
import uvicorn
from fastapi import FastAPI ,Form, HTTPException
from Model import Model
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os




# 2. Create the app object
app = FastAPI()
rfc = pickle.load(open("RFC_model.pkl", "rb"))
symps = pickle.load(open("symptoms.pkl", "rb"))
dis_ind = pickle.load(open("dis_indices.pkl", "rb"))

df2=pd.read_csv("symptom_Description.csv")
df3 = pd.read_csv("precautions_df.csv")

origins = [
    "http://localhost:5173",
]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message



# 5. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted disease
def convertToString(p):
    if isinstance(p, str):  # If p is already a string
        return p
    elif isinstance(p, (int, float)):  # Convert number to string
        return str(p)
    else:
        print(p)
        return str(p)  # Convert other data types to string

def function(disease):
    d = df2[df2["Disease"] == disease]
    if not d.empty:
        details = d["Description"].values[0]  # Extract the first value
        des = convertToString(details)
    else:
        des = ""

    p_ = df3[df3["Disease"] == disease]
    
    p_1 = convertToString(p_["Precaution_1"].values[0]) if not p_["Precaution_1"].empty and isinstance(p_["Precaution_1"].values[0], str) else ""
    p_2 = convertToString(p_["Precaution_2"].values[0]) if not p_["Precaution_2"].empty and isinstance(p_["Precaution_2"].values[0], str) else ""
    p_3 = convertToString(p_["Precaution_3"].values[0]) if not p_["Precaution_3"].empty and isinstance(p_["Precaution_3"].values[0], str) else ""
    p_4 = convertToString(p_["Precaution_4"].values[0]) if not p_["Precaution_4"].empty and isinstance(p_["Precaution_4"].values[0], str) else ""

    return des, p_1, p_2, p_3, p_4


@app.post('/disease')
def predict(data: Model):
    symps_dict = dict.fromkeys(symps, 0)

    # Update symptoms dictionary based on given_symps
    given_symps = data.given_symps
    for symptom in given_symps:
        if symptom in symps_dict:
            symps_dict[symptom] = 1

    lt = list(symps_dict.values())
    arr = np.array(lt)
    if len(arr) > 132:
        return {'error': 'Wrong input'}
    arr = arr.reshape(1, -1)
    prediction = dis_ind[rfc.predict(arr)[0]]
    print(prediction)
    des,p_1,p_2,p_3,p_4 = function(prediction)
    l = []
    if p_1:
        l.append(p_1)
    if p_2:
        l.append(p_2)
    if p_3:
        l.append(p_3)
    if p_4:
        l.append(p_4)
    return {'prediction': prediction ,'description':des,'precaution':l}



# 6. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)


# if __name__ == '__main__':
#       import uvicorn
#       uvicorn.run(app)
