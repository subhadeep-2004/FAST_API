from pydantic import BaseModel, Field
import pickle as pkl

# Load the symptoms from the file
symps = pkl.load(open("symptoms.pkl", "rb"))

# Define the BaseModel class with the symptoms list
class Model(BaseModel):
    given_symps: list[str]
