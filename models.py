# backend/models.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("❌ MONGODB_URI is missing from your .env file!")
client = MongoClient(MONGO_URI)

# FIX: Use strictly lowercase to satisfy MongoDB's case-sensitivity rules
db = client.entry_shield

# Note: Data will be saved inside the "gate_passes" collection!
visits_collection = db.gate_passes

try:
    visits_collection.create_index("studentId", unique=True, sparse=True)
    print("✅ AWS MongoDB Connected. Indexes verified.")
except Exception as e:
    print(f"⚠️ Index Notice: {e}")