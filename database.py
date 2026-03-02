import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Connect to AWS MongoDB Atlas
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

# Database and Collection names from your mongo-init files
db = client.entry_shield
visits_collection = db.gate_passes

def get_db():
    return visits_collection