import requests
import json 

post1= {
  "age": 39,
  "workclass": "State-gov",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education_num": 13,
  "marital_status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital_gain": 2174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}
response1 = requests.post('https://censussalary.herokuapp.com/predict/', data=json.dumps(post1))


post2= {
  "age": 23,
  "workclass": "Self-emp-inc",
  "fnlgt": 76516,
  "education": "Bachelors",
  "education_num": 13,
  "marital_status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital_gain": 0,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United States"
}
response2 = requests.post('https://censussalary.herokuapp.com/predict/', data=json.dumps(post2))


print(response1.status_code)
print(response1.json())
print("-----------")
print(response2.status_code)
print(response2.json())