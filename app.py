import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
 
model = load_model(r"Model/extree_model_20220727")
 

@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def handle_file():
    
    with st.expander("Click to see the format of the CSV file"):
        st.table(features)
    uploaded_file = st.file_uploader("Please Upload CSV file", type="csv") 
    if uploaded_file is not None:
        st.success("File Uploaded successfully")
        submit = st.button("Predict",help="Click to get prediction")
        if submit:
            with st.spinner("Please wait...."):     
                try:
                    data = pd.read_csv(uploaded_file)
                    df = pd.DataFrame(data,columns = features)
                    predictions = predict_model(model, df)
                    value_count = predictions['Label'].value_counts()
                    st.write(value_count)
                    st.info("Check the Label column for the predicted value")
                    st.write(predictions)
                    csv = convert_df(predictions)
                    st.download_button(label="Download the results as CSV file",
                                       data=csv,
                                       file_name="results.csv",
                                       mime='text/csv')
                except Exception as e:
                    st.exception(e)
                    st.error("Error in Analyzing the file. Please check the attribute names")             
    else: 
        st.error("No file uploaded")
        

# Creating options list for dropdown menu

options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]             
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
options_sex = ['Male','Female','Unknown']
options_education = ['Elementary school','Junior high school',  'High school', 'Above high school','Illiterate', 'Writing & reading','Unknown']
options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']
options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']
options_owner = ['Owner','Governmental', 'Organization', 'Other']
options_service_year = ['5-10yrs','2-5yrs', 'Above 10yr', 'Unknown', '1-2yr',
       'Below 1yr']
options_accident_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']
options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
       'other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)', 'Unknown']
options_road_type = ['Tangent road with flat terrain',
 'Tangent road with mild grade and flat terrain',
 'Steep grade downward with mountainous terrain',
 'Tangent road with mountainous terrain and', 'Escarpments',
 'Steep grade upward with mountainous terrain', 'Sharp reverse curve',
 'Gentle horizontal curve' 'Tangent road with rolling terrain']
options_junc_type = ['No junction', 'Y Shape', 'Crossing', 'Other', 'Unknown', 'O Shape', 'T Shape'
 'X Shape']
options_road_surface = ['Asphalt roads','Earth roads', 
 'Asphalt roads with some distress', 'Gravel roads', 'Other']
options_road_condition =['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep']
options_light_condition = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
 'Darkness - lights unlit']
options_weather_condition = ['Normal', 'Raining', 'Raining and Windy', 'Windy', 'Cloudy', 'Snow', 
 'Fog or mist', 'Other', 'Unknown']
options_collision_type = ['Vehicle with vehicle collision',
 'Collision with roadside-parked vehicles', 'Collision with animals',
 'Collision with roadside objects', 'Collision with pedestrians',
 'With Train', 'Rollover', 'Fall from vehicles',  'Other', 'Unknown']
options_vehicle_movement = ['Going straight', 'U-Turn', 'Waiting to go', 'Moving Backward', 'Reversing',
 'Turnover', 'Parked', 'Stopping', 'Getting off',
 'Overtaking','Entering a junction','Other','Unknown']
options_casualty_class = ['Driver or rider', 'Passenger', 'Pedestrian']
options_casualty_severity = ['Slight', 'Serious', 'Fatal']
options_causaly_map = {'Slight': 1, 'Serious': 2, 'Fatal': 3, 'Unknown': None}
options_pedestrian_movement = ['Not a Pedestrian', "Crossing from driver's nearside",
 'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle',
 'Walking along in carriageway, back to traffic',
 'Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle',
 'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)',
 'Walking along in carriageway, facing traffic',
 'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle', 'Unknown or other']
options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']

features = ['Day_of_week', 'Age_band_of_driver', 'Sex_of_driver',
       'Educational_level', 'Driving_experience', 'Type_of_vehicle',
       'Owner_of_vehicle', 'Service_year_of_vehicle', 'Area_accident_occured',
       'Lanes_or_Medians', 'Road_allignment', 'Types_of_Junction',
       'Road_surface_type', 'Road_surface_conditions', 'Light_conditions',
       'Weather_conditions', 'Type_of_collision',
       'Number_of_vehicles_involved', 'Number_of_casualties',
       'Vehicle_movement', 'Casualty_class', 'Sex_of_casualty',
       'Age_band_of_casualty', 'Casualty_severity', 'Pedestrian_movement',
       'Cause_of_accident', 'Hour', 'Minute']



def handle_form():
    with st.form('prediction_form'):
        st.subheader("Enter the input for following features:")
        
        Day_of_week = st.selectbox("Select Accident Day: ", options=options_day)
        accident_time = st.time_input('Select Accident Time')
        Hour = accident_time.hour
        Minute = accident_time.minute
        Age_band_of_driver = st.selectbox("Select Driver Age: ", options=options_age)
        Sex_of_driver = st.selectbox("Select Driver Gender: ", options=options_sex)
        Educational_level = st.selectbox("Select Driver Education: ", options=options_education)
        Driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
        Type_of_vehicle =  st.selectbox("Select Vehicle Type: ", options=options_vehicle_type)
        Owner_of_vehicle = st.selectbox("Select Vehicle Belonging: ", options=options_owner)
        Service_year_of_vehicle = st.selectbox("Select Vehicle Service Year: ", options=options_service_year)
        Area_accident_occured = st.selectbox("Select Accident Area: ", options=options_accident_area)
        Lanes_or_Medians = st.selectbox("Select Lanes: ", options=options_lanes)
        Road_allignment = st.selectbox("Select Road Allignment: ", options=options_road_type)
        Types_of_Junction = st.selectbox("Select Junction Type: ", options=options_junc_type)
        Road_surface_type = st.selectbox("Select Road Surface Type: ", options=options_road_surface)
        Road_surface_conditions = st.selectbox("Select Road Surface Conditions: ", options=options_road_condition)
        Light_conditions = st.selectbox("Select Light Conditions: ", options=options_light_condition)
        Weather_conditions = st.selectbox("Select Weather Conditions: ", options=options_weather_condition)
        Type_of_collision = st.selectbox("Select Collision Type: ", options=options_collision_type)
        Number_of_vehicles_involved = st.slider("Number of Vehicles Involved: ", 1, 7, value=1, format="%d")
        Number_of_casualties = st.slider("Number of Casualties: ", 1, 8, value=1, format="%d")
        Vehicle_movement = st.selectbox("Select Vehicle Movement: ", options=options_vehicle_movement)
        Casualty_class = st.selectbox("Select Casualty Class: ", options=options_casualty_class)
        Sex_of_casualty = st.selectbox("Select Casualty Gender: ", options=options_sex)
        if Sex_of_casualty == "Unknown":
            Sex_of_casualty = None
        Age_band_of_casualty = st.selectbox("Select Casualty Age: ", options=options_age)
        if Age_band_of_casualty == "Unknown":
            Age_band_of_casualty = None
        Casualty_severity = st.selectbox("Select Casualty Severity: ", options=options_casualty_severity)
        Casualty_severity = options_causaly_map[Casualty_severity]
        Pedestrian_movement = st.selectbox("Select Pedestrian Movement: ", options=options_pedestrian_movement)
        Cause_of_accident = st.selectbox("Select Cause of Accident: ", options=options_cause)
       
        
        
        submit = st.form_submit_button("Predict")
        
        if submit:
            data =  np.array([Day_of_week,Age_band_of_driver,Sex_of_driver,Educational_level,Driving_experience,Type_of_vehicle,
                              Owner_of_vehicle,Service_year_of_vehicle,Area_accident_occured,
                              Lanes_or_Medians,Road_allignment,Types_of_Junction,Road_surface_type,
                              Road_surface_conditions,Light_conditions,
                              Weather_conditions,Type_of_collision,Number_of_vehicles_involved,
                              Number_of_casualties,Vehicle_movement,Casualty_class,Sex_of_casualty,Age_band_of_casualty,
                              Casualty_severity,Pedestrian_movement,Cause_of_accident,Hour,Minute]).reshape(1,-1)
            
            data = pd.DataFrame(data,columns = features)
            
            
            prediction = predict_model(model, data)["Label"][0]
            
            st.markdown(f"<h3 style='color:blue;'>The Predicted Severity is: {prediction} </h3>", unsafe_allow_html=True)
                
    

def main():
    
    st.set_page_config(page_title="Accident Severity Prediction App", page_icon="🚘💥🚗", layout="wide")

    # Creating options for list for dropdown menu

    st.markdown("<h1 style='text-align: center;color:medium-blue;'>Accident Severity Prediction App 🚘💥🚗</h1>", unsafe_allow_html=True)

    st.markdown(
    """ <style>
            div[role="radiogroup"] >  :first-child{
                display: none !important;
            }
        </style>
        """,
    unsafe_allow_html=True
    )

    st.subheader("Input data source")
    data_option = st.radio(label="Choose one of the following options",options = ["-", "Form data", "CSV file"])

    st.sidebar.info("This app is developed to classify the severity of the accident based on the data provided by the user")
    st.sidebar.image("Static/car-crash.jpg")
    st.sidebar.success("Made with ❤️ by [Hrithik-Kumar](https://www.linkedin.com/in/hrithik-k-586967141/) ")         
    if data_option == "CSV file":
        handle_file()            
    elif data_option == "Form data":
        handle_form()
        
           
if __name__ == '__main__':
    main()