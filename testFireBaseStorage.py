import numpy as np

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://pbl5-94125-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "pbl5-94125.appspot.com"
})

bucket = storage.bucket()

folderPath = 'Image'
blobs = bucket.list_blobs(prefix=folderPath)

for blob in blobs:
    print(blob.name)
    print(blob.name.split('/')[1])
