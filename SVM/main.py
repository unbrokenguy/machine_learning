import pandas as pd
import numpy as np


def rand_symptoms(n):
    r = np.random.randint(0, 2, n)
    return r


def get_disease_probability(disease_column, person_symptoms):
    probability = 1
    for i in range(len(person_symptoms)):
        if person_symptoms[i] == 1:
            probability *= disease_column[i] * person_symptoms[i]
    return probability


def get_all_disease_probabilities_for_person(
    symptoms, person_symptoms, diseases_probabilities
):
    all_diseases = symptoms.columns[1:]
    symptoms_prob = np.ones(symptoms.shape[1] - 1)
    for i in range(len(all_diseases)):
        symptoms_prob[i] = (
            get_disease_probability(symptoms[all_diseases[i]], person_symptoms)
            * diseases_probabilities[i]
        )
    return symptoms_prob


if __name__ == "__main__":
    symptoms_data = pd.read_csv("symptom.csv", delimiter=";")
    diseases_data = pd.read_csv("disease.csv", delimiter=";")
    diseases_prob = (
        diseases_data["количество пациентов"]
        / list(diseases_data["количество пациентов"])[-1]
    )
    diseases_prob = list(diseases_prob)[:-1]
    diseases_probabilities = list(
        diseases_data["количество пациентов"]
        / list(diseases_data["количество пациентов"])[-1]
    )
    del diseases_probabilities[-1]
    symptoms_number = symptoms_data.shape[0]
    person = rand_symptoms(symptoms_number)
    diseases_probabilities = get_all_disease_probabilities_for_person(
        symptoms_data, person, diseases_probabilities
    )
    print(symptoms_data.columns[np.argmax(diseases_probabilities) + 1])
