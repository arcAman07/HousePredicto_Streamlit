import streamlit as st
import pickle
import pandas as pd
import numpy as np

Data = pd.read_csv("dataset/Boston.csv")


pickle_in = open("C:/Users/amans/housing.pkl", "rb")
classifier = pickle.load(pickle_in)


def mood_recog(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat):
    prediction = classifier.predict(
        [[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]]
    )
    print(prediction)
    return prediction


def main():
    st.title("BostonHousing ML Web App")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Menstruation Cycle Predictor </h2>
    </div>
    """
    crim = st.text_input("Enter per capita crime rate by town: ")
    zn = st.text_input(
        "Enter proportion of residential land zoned for lots over 25,000 sq.ft: "
    )
    indus = st.text_input("Enter proportion of non-retail business acres per town: ")
    chas = st.text_input(
        "Enter Charles River dummy variable (= 1 if tract bounds river; 0 otherwise): "
    )
    nox = st.text_input("Enter Nitric oxides concentration (parts per 10 million): ")
    rm = st.text_input("Enter average number of rooms per dwelling: ")
    age = st.text_input(
        "Enter Proportion of owner-occupied units built prior to 1940: "
    )
    dis = st.text_input("Enter Weighted distances to five Boston employment centres: ")
    rad = st.text_input("Enter Index of accessibility to radial highways: ")
    tax = st.text_input("Enter Full-value property-tax rate per $10,000: ")
    ptratio = st.text_input("Enter Pupil-teacher ratio by town: ")
    b = st.text_input(
        "Enter 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town: "
    )
    lstat = st.text_input("Enter % lower status of the population: ")
    if st.button("Predict"):
        result = mood_recog(
            crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat
        )
        st.success(
            "The Median value of owner-occupied homes in 1000's of dollars is {} ".format(
                result[0]
            )
        )


if __name__ == "__main__":
    main()
