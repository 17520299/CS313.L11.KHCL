import streamlit as st
import pandas as pd
import numpy as np
import Preprocessing as pre
import pickle

#choose data
def user_input_features():
    Gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    CustomerType = st.sidebar.selectbox('Customer Type', ('Loyal Customer', 'disloyal Customer'))
    Age = st.sidebar.slider('Age', 7, 85)
    TypeofTravel = st.sidebar.selectbox('Type of Travel', ('Personal Travel', 'Business travel'))
    Class = st.sidebar.selectbox('Class', ('Eco Plus', 'Business'))
    FlightDistance = st.sidebar.slider('FlightDistance', 31, 4983)
    Inflight_wifi_service = st.sidebar.slider('Inflight wifi service', 1,5)
    Departure_Arrival_time_convenient = st.sidebar.slider('Departure/Arrival time convenient',1,5)
    Gatelocation = st.sidebar.slider('Gate location', 1,5)
    Ease_of_Online_booking = st.sidebar.slider('Ease of Online booking', 1,5)
    Food_and_drink = st.sidebar.slider('Food and drink', 1,5)
    Onlineboarding = st.sidebar.slider('Online boarding', 1,5)
    Seat_comfort = st.sidebar.slider('Seat comfort', 1,5)
    Inflight_entertainment = st.sidebar.slider('Inflight entertainment', 1,5)
    On_board_service = st.sidebar.slider('On-board service', 1,5)
    Leg_room_service = st.sidebar.slider('Leg room service', 1,5)
    Baggage_handling = st.sidebar.slider('Baggage handling', 1,5)
    Checkin_service = st.sidebar.slider('Checkin service', 1,5)
    Inflight_service = st.sidebar.slider('Inflight service', 1,5)
    Cleanliness = st.sidebar.slider('Cleanliness', 1,5)
    Arrival_Delay_in_Minutes = st.sidebar.slider('Arrival Delay in Minutes', 6, 18, 23)
    Departure_Delay_in_Minutes = st.sidebar.slider('Departure Delay in Minutes', 0, 109)
    datacl = {'Gender': Gender,
              'Customer Type': CustomerType,
              'Age': Age,
              'Type of Travel': TypeofTravel,
              'Class': Class,
              'Flight Distance': FlightDistance,
              'Inflight wifi service': Inflight_wifi_service,
              'Departure/Arrival time convenient': Departure_Arrival_time_convenient,
              'Ease of Online booking': Ease_of_Online_booking,
              'Gate location': Gatelocation,
              'Food and drink': Food_and_drink,
              'Online boarding': Onlineboarding,
              'Seat comfort': Seat_comfort,
              'Inflight entertainment': Inflight_entertainment,
              'On-board service': On_board_service,
              'Leg room service': Leg_room_service,
              'Baggage handling': Baggage_handling,
              'Checkin service': Checkin_service,
              'Inflight service': Inflight_service,
              'Cleanliness': Cleanliness,
              'Arrival Delay in Minutes': Arrival_Delay_in_Minutes,
              'Departure Delay in Minutes': Departure_Delay_in_Minutes
              }
    features = pd.DataFrame(datacl, index=[0])
    return features

st.set_option('deprecation.showfileUploaderEncoding', False)
features = st.sidebar.selectbox('Features',('Raw Features','PCA'))
st.sidebar.subheader("Visualization Setting")
uploaded_file = st.sidebar.file_uploader(label="Upload your CSV", type=['csv'])

#load saved classification model
logis = pickle.load(open('C:/Users/Admin/PycharmProjects/pythonProject/logisraw.pkl','rb'))
logis_pca = pickle.load(open('C:/Users/Admin/PycharmProjects/pythonProject/logisPCA.pkl','rb'))
if uploaded_file is None:
    data = user_input_features()
    dataraw = pd.read_csv('data_clean.csv')
    dataraw = pre.ReadNDrop(dataraw)
    data = pd.concat([data, dataraw], axis=0)
    data = pre.TranformImport(data)
    data_pca = pre.PCAImport(data)
    data = data[:1]
    data_pca = data_pca[:1]
else:
    data = pd.read_csv(uploaded_file)
    data = pre.ReadNDrop(data)
    data = pre.TranformImport(data)
    data_pca = pre.PCAImport(data)
st.write(data)
st.write(data_pca)
if features == 'Raw Features':
    prediction_proba = logis.predict_proba(data)
elif features == 'PCA':
    prediction_proba = logis_pca.predict_proba(data_pca)

st.write(prediction_proba)