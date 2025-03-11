import requests

url = "http://127.0.0.1:5000/predict"
file = {'file': open(r"C:\Users\user\mon_projet_reconnaissance\data\split_ttv_dataset_type_of_plants\Test_Set_Folder\aloevera\aloevera42.jpg", "rb")}

response = requests.post(url, files=file)
print(response.json())  # Affiche la prédiction
