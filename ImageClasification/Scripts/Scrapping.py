import pandas as pd
import requests

url = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"

r = requests.get(url)

#See which data I can download e
with open("class-descriptions-boxable.csv", "wb") as f:
    f.write(r.content)

print("Downloaded!")


classes = pd.read_csv("class-descriptions-boxable.csv", header=None)
classes.columns = ["ClassID", "Name"]

print(classes.head())