# Road Traffic Accidents Severity Prediction App
## Project: Predicting the Road Traffic Accident Severity


<!-- ## App: https://rta-streamlit.herokuapp.com/ -->

<br/>

![Alt text](Static/visualization.jpg?raw=true "Title")

## Dataset Description

The data set is collected from Addis Ababa Sub-city police departments for master's research work. The data set has been prepared from manual records of road traffic accidents of the year 2017-20. All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident. Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms.

Source of Dataset: [Click here](https://www.narcis.nl/dataset/RecordID/oai%3Aeasy.dans.knaw.nl%3Aeasy-dataset%3A191591) 
## Probelm Statement

The target feature is **Accident_severity** which is a multi-class variable. The task is to classify this variable based on the other 31 features step-by-step by going through each day's task. The metric used for evaluation is **f1-score**

<br/>

## Install

This project requires **Python** and the following Python Libraries installed:

### RTSA Notebook

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [plotly](https://plotly.com/)
- [imblearn](https://imbalanced-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
  
Following two packages are optional

- [Shap](https://shap.readthedocs.io/en/latest/index.html) (Required only for Explainable AI)
- [Joblib](https://joblib.readthedocs.io/en/latest/) (Required only for Saving and loading the model)



### RTSA_Pycaret Notebook

- [Pycaret](https://pycaret.org/) 
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

<br/>

- [Streamlit](https://streamlit.io/) (Only required to run the web application)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html) or you can use [Google Collab](https://colab.research.google.com/)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included. 

### Code

Python Notebooks are provided in the `Notebook` folder. The required datasets are included in the `Dataset` Folder. 

### Running the Python Notebooks Locally

In a terminal or command window, navigate to the top-level project directory `RTA-PROJECT/` (that contains this README) and then navigate to `Notebook` and run one of the following commands:

```bash
ipython notebook "Notebook_name"
```  
or
```bash
jupyter notebook "Notebook_name"
```
or open with Juoyter Lab
```bash
jupyter lab
```

This will open the Jupyter Notebook software and project file in your browser.

## Running the Application Locally

In a terminal or command window, navigate to the top-level project directory `RTA-PROJECT/` (that contains this README) and run the following command:

```bash
streamlit run app.py
```  











