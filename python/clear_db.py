import os
from pymongo import MongoClient

# --- Configuration ---
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "banqi_training"
COLLECTION_NAME = "games"

def clear_database():
    print(f"Connecting to MongoDB at {MONGO_URI}...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    # Check if collection exists
    if COLLECTION_NAME in db.list_collection_names():
        count = db[COLLECTION_NAME].count_documents({})
        print(f"Found {count} documents in '{DB_NAME}.{COLLECTION_NAME}'.")
        
        confirmation = input("Are you sure you want to delete all documents? (yes/no): ")
        if confirmation.lower() == 'yes':
            db[COLLECTION_NAME].drop()
            print(f"Collection '{COLLECTION_NAME}' dropped.")
        else:
            print("Operation cancelled.")
    else:
        print(f"Collection '{COLLECTION_NAME}' does not exist in database '{DB_NAME}'.")

if __name__ == "__main__":
    clear_database()
