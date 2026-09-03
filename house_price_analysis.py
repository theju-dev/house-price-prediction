import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import joblib
df=pd.read_csv("data/house_prices.csv")
print("\n First 5 rows")
print(df.head(5))
print("\n Dataset shape")
print(df.shape)
print("\nColumn names")
print(df.columns)
print("\n Data types and null values")
df.info()
print("\n Missing values")
print(df.isnull().sum())
print("\n Duplicate rows")
print(df.duplicated().sum())
print("\n Numerical columns")
print(df.describe())
## DATA CLEANING ##
df=df.drop(columns=["Dimensions","Plot Area"])
df.dropna(subset=["Price (in rupees)"])
print(df.shape)
print(df.isnull().sum())
print("\n PRICE DISTRIBUTION SUMMARY")
print(df["Price (in rupees)"].describe())
print("\n Lowest 10 prices")
print(df[["Title","location","Price (in rupees)"]].sort_values("Price (in rupees)").head(10))
print("\n Highest 10 prices")
print(df[["Title","location","Price (in rupees)"]].sort_values("Price (in rupees)",ascending=False).head(10))
zero_price_count=(df["Price (in rupees)"]==0).sum()
print("\nNumber of zero-price properties")
print(zero_price_count)
zero_price_percentage=(zero_price_count/df["Price (in rupees)"].notna().sum())*100
print("zero_price_percentage:")
print(round(zero_price_percentage,2),"%")
#remove invalid target values
df=df[df["Price (in rupees)"]>0].copy()
print(df.shape)
print(df["Price (in rupees)"].min())
print("\ncolumn data types")
print(df.dtypes)
print("\nunique values per column")
for column in df.columns:
    print(column,":",df[column].nunique())
candidate_columns=["Carpet Area","Bathroom","Balcony","Car Parking","Super Area","Furnishing","Transaction","Status"]
for column in candidate_columns:
    print(f"\n Sample values - {column}")
    print(df[column].dropna().unique()[:20])
df["Carpet Area Numeric"]=df["Carpet Area"].str.extract(r"(\d+\.?\d*)")[0].astype(float)
print(df["Carpet Area Numeric"])
print(df[["Carpet Area","Carpet Area Numeric"]].head(20))
print("\n Missing carpet area numeric:")
print(df["Carpet Area Numeric"].isnull().sum())
print("\n Bathroom value counts")
print(df["Bathroom"].value_counts(dropna=False).head(20))
df["Bathroom Numeric"]=df["Bathroom"].str.extract(r"(\d+)")[0].astype(float)
print(df["Bathroom Numeric"])
##df.loc[rows,columns]
df.loc[df["Bathroom"]=="> 10","Bathroom Numeric"]=11
print("\n Bathroom cleaning check")
print(df[["Bathroom","Bathroom Numeric"]].head(20))
print("\n Missing bathroom numeric")
print(df["Bathroom Numeric"].isnull().sum())
print("\n Balcony value counts")
print(df["Balcony"].value_counts(dropna=False).head(20))
df["Balcony Numeric"]=df["Balcony"].str.extract(r"(\d+)")[0].astype(float)
print(df[["Balcony","Balcony Numeric"]].head(10))
print(df["Balcony Numeric"].isnull().sum())
df.loc[df["Balcony"]=="> 10","Balcony Numeric"]=11
print(df.loc[df["Balcony"]=="> 10","Balcony Numeric"])
print("\n Car parking value counts")
print(df["Car Parking"].value_counts(dropna=False))
df["Car Parking Count"]=df["Car Parking"].str.extract(r"(\d+)")[0].astype(float)
df["Car Parking Type"]=df["Car Parking"].str.extract(r"(Open|Covered)")[0]
print("\n Car Parking cleaning check")
print(df[["Car Parking","Car Parking Type","Car Parking Count"]].head(20))
print("\n Missing car parking count")
print(df["Car Parking Count"].isnull().sum())
print("\n Missing car parking type")
print(df["Car Parking Type"].isnull().sum())
print("\n Super Area value counts")
print(df["Super Area"].value_counts(dropna=False).head(20))
df["Super Area Numeric"]=df["Super Area"].str.extract(r"(\d+\.?\d*)")[0].astype(float)
print(df[["Super Area","Super Area Numeric"]].head(20))
print("\n Missing super area numeric")
print(df["Super Area Numeric"].isnull().sum())
print("\n Categorical feature check")
categorical_columns=["Furnishing","Transaction","Status","Car Parking Type"]
for column in categorical_columns:
    print(f"\n Value counts of {column}")
    print(df[column].value_counts(dropna=False))
print("\n Transaction counts")
print(df["Transaction"].value_counts(dropna=False))
print("\n PRICE SUMMARY BY TRANSACTION")
print(df.groupby("Transaction")["Price (in rupees)"].agg(["count","mean","median","min","max"]))
df=df[df["Transaction"].isin(["Resale","New Property"])].copy()
print("\n Transaction after cleaning")
print(df["Transaction"].value_counts(dropna=False))
print("\n shape after transaction cleaning")
print(df.shape)
model_features=["Carpet Area Numeric","Bathroom Numeric","Balcony Numeric","Car Parking Count",
                "Super Area Numeric","Furnishing","Transaction","Car Parking Type"]
print("\n Missing values in model features")
for column in model_features:
    print(column,":",df[column].isnull().sum())
numeric_features=["Carpet Area Numeric","Bathroom Numeric","Balcony Numeric","Car Parking Count","Super Area Numeric"]
categorical_features=["Furnishing","Transaction","Car Parking Type"]
X=df[numeric_features + categorical_features]
y=df["Price (in rupees)"]
print("\n X shape")
print(X.shape)
print("\n Y shape")
print(y.shape)
print(X.head())
print(y.head())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("\n Training shapes")
print("X_train:",X_train.shape)
print("y_train:",y_train.shape)
print("\n Testing shapes")
print("X_test:",X_test.shape)
print("y_test:",y_test.shape)
numeric_transformer=Pipeline(steps=[("imputer",SimpleImputer(strategy="median"))])
categorical_transformer=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                                        ("encoder",OneHotEncoder(handle_unknown="ignore"))])
preprocessor=ColumnTransformer(transformers=[("num",numeric_transformer,numeric_features),
                               ("cat",categorical_transformer,categorical_features)])
model=Pipeline(steps=[("preprocessor",preprocessor),
                      ("regressor",LinearRegression())])
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("X_train:",X_train)
print("y_train:",y_train)
print("X_test:",X_test)
print("y_test:",y_test)
comparision=pd.DataFrame({"Actual Price": y_test.to_numpy(),
              "Predicted Price" : y_pred})
print(comparision)
mae=mean_absolute_error(y_test,y_pred)
print("\n Model evaluation")
print("MAE:",mae)
mse=mean_squared_error(y_test,y_pred)
rmse=mse**0.5
r2=r2_score(y_test,y_pred)
print("\n Model Evaluation")
print("MAE:",mae)
print("RMSE:",rmse)
print("r^2:",r2)
comparision["Error"]=comparision["Actual Price"]-comparision["Predicted Price"]
comparision["Absolute Error"]=comparision["Error"].abs()
worst_predictions=comparision.sort_values("Absolute Error",ascending=False)
print(worst_predictions.head(20))
print("\n Price percentiles")
print(df["Price (in rupees)"].quantile([0.50,0.75,0.90,0.95,0.99,0.995,0.999,1.0]))
extreme_prices=df[df["Price (in rupees)"]>50000]
print("\n Extreme price rows")
print(extreme_prices[["Price (in rupees)","Amount(in rupees)","location","Carpet Area","Super Area","Title"]].sort_values("Price (in rupees)",ascending=False).head(30))
print("\n Extreme price counts")
print("Above 50000:",(df["Price (in rupees)"]>50000).sum())
print("Above 100000:",(df["Price (in rupees)"]>100000).sum())
print("Above 500000:",(df["Price (in rupees)"]>500000).sum())
print("Above 1000000:",(df["Price (in rupees)"]>1000000).sum())
extreme_count=(df["Price (in rupees)"] > 50000).sum()
extreme_percentage=(extreme_count/len(df))*100
print("Percentage above 50000:",extreme_percentage)
model_df=df[df["Price (in rupees)"]<=50000].copy()
X_model2=model_df[numeric_features + categorical_features]
y_model2=model_df["Price (in rupees)"]
X_train2,X_test2,y_train2,y_test2=train_test_split(X_model2,y_model2,test_size=0.2,random_state=42)
model2=Pipeline(steps=[("Preprocessor",preprocessor),
                       ("regressor",LinearRegression())])
model2.fit(X_train2,y_train2)
y_pred2=model2.predict(X_test2)
mae2=mean_absolute_error(y_test2,y_pred2)
mse2=mean_squared_error(y_test2,y_pred2)
rmse2=mse2**0.5
r2_2=r2_score(y_test2,y_pred2)
print("\n Model2 evalaution")
print("mae:",mae2)
print("rmse:",rmse2)
print("r^2:",r2_2)
print("\n Location analysis")
print("Unique locations")
print(model_df["location"].nunique())
print("\n Top 20 locations")
print(model_df["location"].value_counts().head(20))
categorical_features_model3=["Furnishing","Transaction","Car Parking Type","location"]
X_model3=model_df[numeric_features + categorical_features_model3]
y_model3=model_df["Price (in rupees)"]
preprocessor_model3=ColumnTransformer(transformers=[("num",numeric_transformer,numeric_features),
                                                    ("cat",categorical_transformer,categorical_features_model3)])
X_model3=model_df[numeric_features + categorical_features_model3]
y_model3=model_df["Price (in rupees)"]
X_train3,X_test3,y_train3,y_test3=train_test_split(X_model3,y_model3,test_size=0.2,random_state=42)
model3=Pipeline(steps=[("Preprocessor",preprocessor_model3),("regressor",LinearRegression())])
model3.fit(X_train3,y_train3)
y_pred3=model3.predict(X_test3)
mae3=mean_absolute_error(y_test3,y_pred3)
mse3=mean_squared_error(y_test3,y_pred3)
rmse3=mse3**0.5
r2_3=r2_score(y_test3,y_pred3)
print("\n Model 3 evaluation")
print("\nMAE:",mae3)
print("\nRMSE:",rmse3)
print("\nr^2:",r2_3)
comparision3=pd.DataFrame({"Original index":y_test3.index,
    "Actual Price":y_test3.to_numpy(),
                           "Predicted Price":y_pred3})
comparision3["Absolute error"]=(comparision3["Actual Price"]-comparision3["Predicted Price"]).abs()
print("\n Model3 prediction comparision")
print(comparision3.head(20))
worst_predictions3=comparision3.sort_values(by="Absolute error",ascending=False).head(20)
print("\n Model3 worst predictions")
print(worst_predictions3)
error_analysis3=X_test3.copy()
print("\n X_test3 columns")
print(X_test3.columns.tolist())
error_analysis3["Actual Price"]=y_test3
error_analysis3["Predicted Price"]=y_pred3
error_analysis3["Absolute Error"]=error_analysis3["Actual Price"]-error_analysis3["Predicted Price"].abs()
worst_properties3=error_analysis3.sort_values(by="Absolute Error",ascending=False).head(20)
print("\n Model 3 worst property errors")
print(worst_properties3[["location","Carpet Area Numeric","Bathroom Numeric","Balcony Numeric",
                          "Super Area Numeric","Furnishing","Transaction",
                    "Actual Price","Predicted Price","Absolute Error"]])
joblib.dump(model3,"models/house_price_model.pkl")
loaded_model=joblib.load("models/house_price_model.pkl")
loaded_predictions=loaded_model.predict(X_test3)
print(loaded_predictions[:5])
print(y_pred3[:5])
new_property=pd.DataFrame({"Carpet Area Numeric":[1200],
                           "Bathroom Numeric":[2],
                           "Balcony Numeric":[1],
                           "Car Parking Count":[1],
                           "Super Area Numeric":[1500],
                           "Furnishing":["Semi-Furnished"],
                           "Transaction":["Resale"],
                           "Car Parking Type":["Covered"],
                           "location":["bangalore"]})
new_prediction=loaded_model.predict(new_property)
print("Predicted price:",new_prediction[0])