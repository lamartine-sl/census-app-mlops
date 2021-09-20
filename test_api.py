"""
Test script for API

Author: Lamartine Santana
Date: September 2021
"""
from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

# A function to test the get
def test_get():
    """
    Test get directory
    """
    try:
        response = client.get("/")
    except TypeError:
        print("ERROR: the directory is wrong, please specify the correct path")
    
    validate_content = response.content.decode('utf-8').strip('"')
    assert response.status_code == 200
    assert validate_content == "Hi, Welcome to Census API"


def test_post_less_50k():
    """
    Test the predict output for salary >=50k.
    """
    input_dict = {
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
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    assert json.loads(response.text)["prediction"] == "Salary <= 50k"


def test_post_greater_50k():
    input_dict = {
                    "age": 31,
                    "workclass": "Private",
                    "fnlgt": 45781,
                    "education": "Masters",
                    "education_num": 14,
                    "marital_status": "Never-married",
                    "occupation": "Prof-specialty",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Female",
                    "capital_gain": 14084,
                    "capital_loss": 0,
                    "hours_per_week": 50,
                    "native_country": "United-States"
                    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    assert json.loads(response.text)["prediction"] == "Salary > 50k"