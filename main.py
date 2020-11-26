import streamlit as st
import pandas as pd
import Preprocessing as pre
import pickle
import Visualization as vs
from sklearn.decomposition import PCA
html_heading = '''
# Khai thác dữ liệu và ứng dụng (CS313.L11.KHCL)
## Phân tích dữ liệu về sự hài lòng của hành khách đi máy bay dựa trên đánh giá về 22 tiêu chí.
### Gồm có các chức năng chính:
* Trực quan hóa dữ liệu với biểu đồ Countplot và Catplot.
    * 
* Thực hiện dự báo một dữ liệu mới dưới dạng nhập từ trang web và cả upload file từ máy tính cá nhân.
* Chọn thuộc tính gốc hoặc thuộc tính PCA để thực hiện dự đoán.
* Kết quả các độ đo(Precision, Recall, F1-score) giữa 3 thuật toán máy học(Logistic Regression, Decision Tree, Random Forest).
* Trực quan hóa các thuộc tính quan trọng nhất ảnh hưởng đến nhãn phân lớp của bài toán(Important Features).
'''
st.markdown(html_heading,unsafe_allow_html=True)
st.markdown('<style>h1{color: #FFCF00;}</style>', unsafe_allow_html=True)
st.markdown('<style>h2{color: #EDB200;}</style>', unsafe_allow_html=True)
st.markdown('<style>h3{color: #D01013;}</style>', unsafe_allow_html=True)
st.markdown('<style>li{color: #41898D;}</style>', unsafe_allow_html=True)
st.markdown('<style>h4{color: #CD5C5C;}</style>', unsafe_allow_html=True)
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
raw= pd.read_csv('train.csv')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
features = st.sidebar.selectbox('Features',('Raw Features','PCA'))
mlAlgorithm = st.sidebar.selectbox('Machine Learning Algorithm',('Logistic Regression','Decision Tree','Random Forest'))
st.sidebar.markdown("---")
st.sidebar.subheader("Visualization Setting")
st.sidebar.write("#### Chart with Countplot")
x_label_c=st.sidebar.selectbox("X label",('Gender','Customer Type','Age','Type of Travel','Class','Flight Distance','Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking',
                                'Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service',
                                'Leg room service','Baggage handling','Checkin service','Inflight service','Cleanliness','Arrival Delay in Minutes','Departure Delay in Minutes'))
hue_c=st.sidebar.selectbox("Hue",('Gender','Customer Type','Age','Type of Travel','Class','satisfaction'))
kind_c = st.sidebar.selectbox('Kind Chart',('bar','violin','point'))
Visualize_count = st.sidebar.button(label="Visualize Countplot")
Visualize_cat_c = st.sidebar.button(label='Visualize Catplot 2 attribute')
if Visualize_count:
    st.write('''
    ## Biểu đồ biểu diễn đặc điểm các thuộc tính trong tập dữ liệu
    ''')
    st.pyplot(vs.ChartCountPlot(raw,x_label_c,hue_c))
if Visualize_cat_c:
    st.write('''
        ## Biểu đồ biểu diễn đặc điểm các thuộc tính trong tập dữ liệu
        ''')
    st.pyplot(vs.Chartcatplot2atrribute(raw, x_label_c, hue_c,kind_c))
st.sidebar.write("#### Chart with Catplot")
x_label_cat=st.sidebar.selectbox("X labels",('Gender','Customer Type','Age','Type of Travel','Class','Flight Distance','Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking',
                                'Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service',
                                'Leg room service','Baggage handling','Checkin service','Inflight service','Cleanliness','Arrival Delay in Minutes','Departure Delay in Minutes'))
y_label_cat=st.sidebar.selectbox("Y labels",('Gender','Customer Type','Age','Type of Travel','Class','Flight Distance','Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking',
                                'Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service',
                                'Leg room service','Baggage handling','Checkin service','Inflight service','Cleanliness','Arrival Delay in Minutes','Departure Delay in Minutes'))
hue_cat='satisfaction'
kind_cat = st.sidebar.selectbox('Kind',('bar','violin'))
col_cat=st.sidebar.selectbox("Columns",('Gender','Customer Type','Type of Travel','Class'))
Visualize_cat = st.sidebar.button(label='Visualize Catplot 3 attribute')
if Visualize_cat:
    st.write('#Biểu đồ thống kê sự hài lòng của khách hàng với 3 thuộc tính')
    st.pyplot(vs.ChartCatPlot(raw,x_label_cat,y_label_cat,hue_cat,kind_cat,col_cat))
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(label="Upload your CSV", type=['csv'])
st.sidebar.markdown("---")
#load saved classification model
data_importances = pickle.load(open('data1_importances.pkl','rb'))
logis = pickle.load(open('C:/Users/Admin/PycharmProjects/CS313.L11.KHCL/logisraw.pkl','rb'))
logisPCA = pickle.load(open('C:/Users/Admin/PycharmProjects/CS313.L11.KHCL/logisPCA.pkl','rb'))
tree = pickle.load(open('C:/Users/Admin/PycharmProjects/CS313.L11.KHCL/treeraw.pkl','rb'))
treePCA = pickle.load(open('C:/Users/Admin/PycharmProjects/CS313.L11.KHCL/treePCA.pkl','rb'))
random = pickle.load(open('C:/Users/Admin/PycharmProjects/CS313.L11.KHCL/randomraw.pkl','rb'))
randomPCA = pickle.load(open('C:/Users/Admin/PycharmProjects/CS313.L11.KHCL/randomPCA.pkl','rb'))
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
st.sidebar.markdown("---")
Result = st.sidebar.button(label='Chart Results')
Predict = st.sidebar.button(label='Predict')
Importances = st.sidebar.button(label = 'Importances Features')
if features == 'Raw Features':
    if mlAlgorithm == 'Logistic Regression':
        prediction_proba = logis.predict(data)
    elif mlAlgorithm == 'Decision Tree':
        prediction_proba = tree.predict(data)
    else:
        prediction_proba = random.predict(data)
elif features == 'PCA':
    if mlAlgorithm == 'Logistic Regression':
        prediction_proba = logisPCA.predict(data_pca)
    elif mlAlgorithm == 'Decision Tree':
        prediction_proba = treePCA.predict(data_pca)
    else:
        prediction_proba = randomPCA.predict(data_pca)
# Visualization Result
X_train,X_test,y_train,y_test = pickle.load(open('C:/Users/Admin/PycharmProjects/CS313.L11.KHCL/data_split.pkl','rb'))
algorithm = ['LR','DT','RF']
pca = PCA(n_components=23)
X_test_pca = pca.fit_transform(X_test)
if Result:
    st.pyplot(vs.Chart(logis, tree, random, X_test, y_test, algorithm))
    st.pyplot(vs.Chart(logisPCA, treePCA, randomPCA, X_test_pca, y_test, algorithm))
if Predict:
    st.write(prediction_proba)
if Importances:
    st.pyplot(vs.ChartImportances(tree,random,logis,data_importances))
