import numpy as np
import pandas as pd
import sklearn as sk
import xgboost as xgb
import joblib
import pickle as pk
from sklearn.metrics import r2_score
import os
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from feature_engine.encoding import RareLabelEncoder, MeanEncoder, CountFrequencyEncoder
from feature_engine.datetime import DatetimeFeatures
from feature_engine.outliers import Winsorizer
from feature_engine.selection import SelectBySingleFeaturePerformance
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PowerTransformer, FunctionTransformer
from feature_engine.outliers import Winsorizer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SelectBySingleFeaturePerformance
from feature_engine.encoding import (
	RareLabelEncoder,
	MeanEncoder,
	CountFrequencyEncoder
)

sk.set_config(transform_output="pandas")

#read the training data and splitting
path = os.path.join('data','tr.csv')
tr_df = pd.read_csv(path)
X_tr = tr_df.drop(columns='price')
y_tr = tr_df.price.copy()

#helper functions
# airline
air_trans = Pipeline( steps = [
    ('Grouper', RareLabelEncoder(tol=0.07, replace_with='Others', n_categories=3)),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ] )

# date of journey
feature_to_ext = ['month', 'week', 'day_of_week','day_of_month', 'day_of_year']
doj_trans = Pipeline(steps=[
    ('dt', DatetimeFeatures(features_to_extract=feature_to_ext, yearfirst=True, format='mixed')),
    ('scaler', MinMaxScaler())
])

# source and destination
loc = tr_df.loc[:, ['source', 'destination']]

loc_pipe1 = Pipeline( steps = [
    ('Grouper', RareLabelEncoder(tol=0.08, replace_with='Others', n_categories=3)),
     ('Meanencoder', MeanEncoder()), ('Transformer', PowerTransformer())
        ] )

def is_north(X):
    cols = X.columns.to_list()
    north_reg = ['Delhi','Kolkata','New Delhi']
    return X.assign(**{
        f'{col}_is_in_north': X.loc[:,col].isin(north_reg).astype(int)
        for col in cols
    }).drop(columns=['source','destination'])

loc_trans = FeatureUnion(transformer_list=[('part1', loc_pipe1),('part2',FunctionTransformer(is_north))])

# Dep and Arrival time
time=tr_df.loc[:, ['dep_time', 'arrival_time']]
time_pipe1 = Pipeline(steps=[
    ('dt', DatetimeFeatures(features_to_extract=['hour','minute'])), ('scaler', StandardScaler())
])

def daytime(X):
    cols=X.columns.to_list()
    x_temp = X.assign(**{
        col: pd.to_datetime(X.loc[:,col]).dt.hour
        for col in cols
    })
    return x_temp.assign(**{
        f'{col} daytime':np.select([x_temp.loc[:,col].between(5,12,inclusive='left'),
                                    x_temp.loc[:,col].between(12,16,inclusive='left'),
                                    x_temp.loc[:,col].between(16,20,inclusive='left'),
                                    x_temp.loc[:,col].between(20,23,inclusive='left')],
                                    ['morning','afternoon','evening','night'], default='midnight')
        for col in cols
    }).drop(columns=cols)

time_pipe2 = Pipeline(steps=[
    ('daytime', FunctionTransformer(func = daytime)), ('encoder', CountFrequencyEncoder()),
    ('scaler', StandardScaler())
])
time_trans = FeatureUnion(transformer_list=[('part1', time_pipe1),('part2',time_pipe2)])

# duration
dur = tr_df.loc[:, ['duration']]
dur_pipe1 = Pipeline(steps=[
    ('transformer', PowerTransformer()), ('scaler', StandardScaler())
])

dur_pipe2 = Pipeline(steps=[
    ('outliers', Winsorizer(capping_method='iqr', fold=1.5) ), ('imputer', SimpleImputer(strategy='median'))])

dur_trans = FeatureUnion(transformer_list=[('part1', dur_pipe1),('part2', dur_pipe2)])

# total stops
def is_direct(X):
    return X.assign(is_direct_flight=X.total_stops.eq(0).astype(int))


tot_stops_trans = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),("direct flight", FunctionTransformer(func=is_direct))
])

# additional info
info_pipe1 = Pipeline(steps=[
    ("group", RareLabelEncoder(tol=0.08, n_categories=3, replace_with="Others")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

def have_info(X):
    return X.assign(additional_info=X.additional_info.ne("No Info").astype(int))

info_union = FeatureUnion(transformer_list=[
    ("part1", info_pipe1), ("part2", FunctionTransformer(func=have_info))
])
info_trans = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),("union", info_union)
])

# column transformer
col_trans = ColumnTransformer(transformers=[
    ('air', air_trans, ['airline']), ('doj', doj_trans, ['date_of_journey'] ),
    ('location', loc_trans, ['source', 'destination']), ('time', time_trans, ['dep_time', 'arrival_time']),
    ('duration', dur_trans, ['duration']), ('total_stops',  tot_stops_trans, ['total_stops']),
    ('add_info', info_trans, ['additional_info'])
    ]) 

# feature selector
estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
selector = SelectBySingleFeaturePerformance(estimator=estimator,scoring="r2",threshold=0.08) 

# preprocessor
preprocessor = Pipeline(steps=[
    ("ct", col_trans),("selector", selector)])

#fit and save the preprocessor
preprocessor.fit(X_tr, y_tr)
joblib.dump(preprocessor, 'prepro.joblib')


#streamlit web app
import streamlit as st
st.set_page_config(
    page_title='Flights Fare Prediction App', page_icon='✈️'
)

st.title('Flights Fare Prediction - AWS Sagemaker')
#user inputs
airline = st.selectbox('Airline', options=X_tr.airline.unique())
doj = st.date_input('Date of Journey')
src = st.selectbox('Source', options=X_tr.source.unique())
dest = st.selectbox('Destination', options=X_tr.destination.unique())
dtime = st.time_input('Departure Time')
atime = st.time_input('Arrival Time')
dur = st.number_input('Duration in minutes', step=2)
ts = st.number_input('Total Stops', step=1, min_value=0)
add_info = st.selectbox('Additional Info', options=X_tr.additional_info.unique())

x_new = pd.DataFrame(dict(
    airline=[airline], date_of_journey=[doj], source=[src], 
    destination=[dest], dep_time=[dtime], arrival_time=[atime], 
    duration=[dur], total_stops=[ts], additional_info=[add_info],
)).astype({col:'str' for col in ['date_of_journey', 'dep_time', 'arrival_time']})

#load the model to process the given input on pressing predict
if st.button('Predict'):
    saved_prepro = joblib.load('prepro.joblib')
    x_new_pre = saved_prepro.transform(x_new)

    with open('xgboost-model', 'rb') as f:
        model = pk.load(f)

    x_new_xgb = xgb.DMatrix(x_new_pre)
    pred = model.predict(x_new_xgb)[0]

    st.info(f'The predictd fare is INR {pred:,.0f}')

