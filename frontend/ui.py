import io
import requests
import json
import streamlit as st

# interact with FastAPI endpoint
backend = "http://fastapi:8000"

st.title("Salary Predict based on Census Data")

age = st.number_input("Age/Idade", min_value=18, max_value=99)
workclass = st.selectbox("Workclass/Classe de Trabalho",
                         ['Private', 'Self-emp-not-inc', 'Self-emp-inc',
                          'Federal-gov', 'Local-gov', 'State-gov',
                          'Without-pay', 'Never-worked'])
fnlwgt = st.number_input("fnlwgt", min_value=12285, max_value=1484705)
education = st.selectbox("Education/Educação",
                         ['Bachelors',
                          'Some-college',
                          '11th',
                          'HS-grad',
                          'Prof-school',
                          'Assoc-acdm',
                          'Assoc-voc',
                          '9th',
                          '7th-8th',
                          '12th',
                          'Masters',
                          '1st-4th',
                          '10th',
                          'Doctorate',
                          '5th-6th',
                          'Preschool'])
education_num = st.number_input("Education Number", min_value=1, max_value=16)
marital_status = st.selectbox("Marital Status/ Estado civil",
                              ['Married-civ-spouse',
                               'Divorced',
                               'Never-married',
                               'Separated',
                               'Widowed',
                               'Married-spouse-absent',
                               'Married-AF-spouse'])
occupation = st.selectbox("Occupation/Emprego",
                          ['Tech-support',
                           'Craft-repair',
                           'Other-service',
                           'Sales',
                           'Exec-managerial',
                           'Prof-specialty',
                           'Handlers-cleaners',
                           'Machine-op-inspct',
                           'Adm-clerical',
                           'Farming-fishing',
                           'Transport-moving',
                           'Priv-house-serv',
                           'Protective-serv',
                           'Armed-Forces'])
relationship = st.selectbox("Relationship/Relacionamento",
                            ['Wife',
                             'Own-child',
                             'Husband',
                             'Not-in-family',
                             'Other-relative',
                             'Unmarried'])
race = st.selectbox("Race/Raça",
                    ['White',
                     'Asian-Pac-Islander',
                     'Amer-Indian-Eskimo',
                     'Other',
                     'Black'])
sex = st.selectbox("Sex/Sexo", ['Female', 'Male'])
capital_gain = st.number_input(
    "Capital Gain/Ganho de Capital",
    min_value=0,
    max_value=None)
capital_loss = st.number_input(
    "Capital Loss/Perda de Capital",
    min_value=0,
    max_value=None)
hours_per_week = st.number_input(
    "Hours per Week/Horas de trabalho",
    min_value=0,
    max_value=99)
native_country = st.selectbox("Native Country/País Nativo",
                              ['United-States',
                               'Cuba',
                               'Jamaica',
                               'India',
                               'Mexico',
                               'South',
                               'Puerto-Rico',
                               'Honduras',
                               'England',
                               'Canada',
                               'Germany',
                               'Iran',
                               'Philippines',
                               'Italy',
                               'Poland',
                               'Columbia',
                               'Cambodia',
                               'Thailand',
                               'Ecuador',
                               'Laos',
                               'Taiwan',
                               'Haiti',
                               'Portugal',
                               'Dominican-Republic',
                               'El-Salvador',
                               'France',
                               'Guatemala',
                               'China',
                               'Japan',
                               'Yugoslavia',
                               'Peru',
                               'Outlying-US(Guam-USVI-etc)',
                               'Scotland',
                               'Trinadad&Tobago',
                               'Greece',
                               'Nicaragua',
                               'Vietnam',
                               'Hong',
                               'Ireland',
                               'Hungary',
                               'Holand-Netherlands'])

data = {
    "age": age,
    "workclass": workclass,
    "fnlgt": fnlwgt,
    "education": education,
    "education_num": education_num,
    "marital_status": marital_status,
    "occupation": occupation,
    "relationship": relationship,
    "race": race,
    "sex": sex,
    "capital_gain": capital_gain,
    "capital_loss": capital_loss,
    "hours_per_week": hours_per_week,
    "native_country": native_country
}

if st.button("Predict"):
    response = requests.post(
        backend,
        json=data)
    prediction = response.text
    st.success(f"The prediction from model: {prediction}")
