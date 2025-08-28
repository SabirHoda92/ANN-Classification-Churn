import pandas as pd
#train_test_split from scikit-learn (sklearn) is used to split your dataset into two or more subsets (usually training and testing sets).
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle




df = pd.read_csv("Churn_Modelling.csv")
# print(df.head(4))

#preprocess the data
#drop irrelevant column
#axis=1 → Columns
#Used when you want to drop or apply an operation along columns.
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
# print(df)

#Encode Categorical variables
# LabelEncoder: Converts categorical labels into numeric values
label_encoder_gender = LabelEncoder()

# fit_transform = fit() + transform()
# fit() -> learns parameters (e.g., mean & std for scaling, unique classes for encoding)
# transform() -> applies those learned parameters to the data
# fit_transform() -> does both in one step (used on training data)
df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])

# print(df)

#ONEHOT Encode 'Geography column'
#One hot encoding coverts Text to Vector
one_hot_encode = OneHotEncoder(sparse_output=False)

geo_encoder = one_hot_encode.fit_transform(df[['Geography']])
# print(geo_encoder)

geo_encoder = pd.DataFrame(geo_encoder,columns=one_hot_encode.get_feature_names_out(['Geography']))

# print(geo_encoder)

#Combine one hot encoder column with the original data
df = pd.concat([df.drop('Geography', axis=1), geo_encoder], axis=1)
# print(df.head())

#save the encoders and scaler

with open("label_encoder_gender.pk1", 'wb') as file:
# Pickle is used to save (serialize) Python objects to a file
# and load (deserialize) them back later.
# Commonly used to save trained ML models or preprocessed data
    pickle.dump(label_encoder_gender,file)

with open("one_hot_encode.pk1", 'wb') as file:
    pickle.dump(one_hot_encode, file)


#Divide the dataset into independent and dependent feature
x = df.drop('Exited', axis=1)
y = df['Exited']

#split the data in training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Scale these feature
# StandardScaler is used to standardize features
# It transforms data so that each feature has:
# -> mean = 0
# -> standard deviation = 1
# This ensures all features are on the same scale
# (important for ML models like KNN, SVM, Logistic Regression, Neural Nets)
scaler = StandardScaler()
# fit_transform() -> first learns mean & std from x_train (fit),
# then scales x_train using those values (transform)
x_train = scaler.fit_transform(x_train)
# Only transform() is used (no fit) so that test data
# is scaled using the same mean & std learned from training data
# -> prevents data leakage.
x_test = scaler.transform(x_test)

# print(x_train)

with open("Scaler.pk1", 'wb') as file:
    pickle.dump(scaler,file)



