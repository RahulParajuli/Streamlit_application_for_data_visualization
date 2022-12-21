# from cv2 import dft
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import streamlit as st

# def remove_col(df ,i):
#     df.drop([i], axis = 1,inplace = True)
#     return df

# def column_delete(df, column_name):
#   print("deleting the column: ", column_name)
#   # new_df = (df.drop['column_name'], axis=1)
#   del df[column_name]
#   df.head()
#   return df
 
# def row_delete(df, row_number):
#   print("deleting the row number: ", row_number)
#   df.drop(df.index[row_number])
#   df.head()
#   return df

# def mean_fill(df,column_name):
#   mean_value=df[column_name].mean()
#   filled = df[column_name].fillna(value=mean_value, inplace=True)
#   return filled

# def median_fill(df,column_name):
#   median_value=df[column_name].median()
#   filled = df[column_name].fillna(value=median_value, inplace=True)
#   return filled

# def random_fill(df):
#     for i in df.columns:
#         df[i+"_imputed"] = df[i]
#         df[i+"_imputed"][df[i+"_imputed"].isnull()] = df[i].dropna().sample(df[i].isnull().sum()).values

# def EndDistribution(df, column_name):
    
#       mean = df[column_name].mean()
#       std = df[column_name].std()
#       #calculating extreme standard deviation
#       extreme = (mean + (3*std))
#       df[column_name+'_median'] = df[column_name].fillna(df[column_name].median())
#       df[column_name+'_end_distribution'] = df[column_name].fillna(extreme)
#       return df

# #knn imputer


# def impute_knn(df):
#     '''
#     function for knn imputation in missing values in the data
#     df - dataset provided by the users
#     '''
#     from sklearn.impute import KNNImputer
#     imputer =KNNImputer(n_neighbors=5)
    
#     #finding only numeric columns
#     cols_num = df.select_dtypes(include=np.number).columns
#     for feature in df.columns:
#         #for numeric type
#         if feature in cols_num:   
#             df[feature] = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)))
#         else:
#         #for categorical type
#             df[feature] = df[feature].fillna(df[feature].mode().iloc[0])
#     return df

# #Z score capping 
# def zScore(df):
#     cols_num = df.select_dtypes(include=np.number).columns
#     for i in cols_num:
#         max_threshold = df[i].mean() + 3*df[i].std()
#         min_threshold = df[i].mean() - 3*df[i].std()
# #       df = df[(df['cgpa'] > 8.80) | (df['cgpa'] < 5.11)]
#         df[i] = np.where(
#             df[i]>max_threshold,
#             max_threshold,
#             np.where(
#                 df[i]<min_threshold,
#                 min_threshold,
#                 df[i]
#             )
#         )
#     return df

# # zscore trimming
# def zScore_trim(df):
#     cols_num = df.select_dtypes(include=np.number).columns
#     for i in cols_num:
#         max_threshold = df[i].mean() + 3*df[i].std()
#         min_threshold = df[i].mean() - 3*df[i].std()
#         df = df[(df[i] < max_threshold) | (df[i] > min_threshold)]
#     return df

# # Ourlier using Percentile
# # trimming
# def percentile_trimming(df):
#     cols_num = df.select_dtypes(include=np.number).columns
#     for i in cols_num:
#         percentile25 = df[i].quantile(0.25)
#         percentile75 = df[i].quantile(0.75)
#         iqr = percentile75 - percentile25
#         max_threshold = percentile75 + 3*iqr
#         min_threshold = percentile25 - 3*iqr
#         df = df[(df[i] < max_threshold) | (df[i] > min_threshold)]
#     return df

# #capping
# def percentile_capping(df):
#     cols_num = df.select_dtypes(include=np.number).columns
#     for i in cols_num:
#         percentile25 = df[i].quantile(0.25)
#         percentile75 = df[i].quantile(0.75)
#         iqr = percentile75 - percentile25
#         max_threshold = percentile75 + 3*iqr
#         min_threshold = percentile25 - 3*iqr
#         df[i] = np.where(
#             df[i]>max_threshold,
#             max_threshold,
#             np.where(
#                 df[i]<min_threshold,
#                 min_threshold,
#                 df[i]
#             )
#         )
#     return df

# # Function to find date column in dataframe and convert it to datetime format
# def convert_date(df):
#     '''
#     function parameter  : dataframe
#     parameter datatype  : pandas.core.frame.DataFrame
#     function returns    : dataframe
#     return datatype     : pandas.core.frame.DataFrame
#     function definition : takes dataframe as input and finds the date columns in the dataframe.
#                             if found, converts the column to datetime format.
#     '''
#     df = df.apply(lambda col: pd.to_datetime(col, errors='ignore') if col.dtypes == object else col, axis=0)
#     return df

# # Function to find price column in dataframe
# def price_column(df):
#     '''
#     function parameter  : dataframe
#     parameter datatype  : pandas.core.frame.DataFrame
#     function returns    : dataframe
#     return datatype     : pandas.core.frame.DataFrame
#     function definition : takes dataframe as input and finds the price related columns in the dataframe.
#                             if found, renames the column to price_1.
#     '''
#     numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
#     price_cols = [col for col in numeric_cols if col.lower().find('price') != -1 or col.lower().find('cost') != -1 or 
#                     col.lower().find('total') != -1 or col.lower().find('amount') != -1 or col.lower().find('revenue') != -1 or
#                     col.lower().find('profit') != -1 or col.lower().find('margin') != -1 or col.lower().find('sales') != -1]
#     if len(price_cols) > 1:
#         for i in range(len(price_cols)):
#             df.rename(columns={price_cols[i]: 'price_'+str(i+1)}, inplace=True)
#     elif len(price_cols) == 1:
#         df.rename(columns={price_cols[0]: 'price'}, inplace=True)
#     return df


# def data_cleaning(df):
#     import pandas as pd
#     import numpy as np
#     from sklearn.impute import KNNImputer
#     pd.set_option('display.max_rows', 100)
#     for i in df.columns:
#         if ((df[i].isna().sum())/df.shape[0]) > 0.95:
#             df = remove_col(df,i)
#         else:
#             df = df.copy()
#     df = impute_knn(df)
#     return df


# class missing_df:
#     def __init__(self, df):
#         self.df = df
#         print(self.df)
#functions for handling missing values

class missing_df:
  def __init__ (self,dataset):
    self.dataset = dataset

def handle_missing_value():
    df = pd.read_csv("temp_data/test.csv")
    missing_count = df.isnull().sum().sum()
    if missing_count != 0:
        print(f"Found total of {missing_count} missing values.")

    #remove column having name starts with Unnamed
    df =df.loc[:,~df.columns.str.startswith('Unnamed')]

    #drop columns having more than 90% missing values
    for i in df.columns.to_list():
        if df[f"{i}"].isna().mean().round(4) > 0.9:
            df = df.drop(i, axis=1)
    
    #converting object datatype to integer if present
    for j in df.columns.values.tolist(): # Iterate on columns of dataframe
      try:
          df[j] = df[j].astype('int') # Convert datatype from object to int, of columns having all integer values 
      except:
          pass
    
    
    # find date column in dataframe and convert it to datetime format
    try:
        df = df.apply(lambda col: pd.to_datetime(col, errors='ignore') if col.dtypes == object else col, axis=0)
    except:
        pass

    #impute missing values
    imputer = KNNImputer(n_neighbors=3)
    #finding numerical columns from dataset
    cols_num = df.select_dtypes(include=np.number).columns
    for feature in df.columns:
        #for numeric type
        if feature in cols_num:   
            df[feature] = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)))
        else:
        #for categorical type
            df[feature] = df[feature].fillna(df[feature].mode().iloc[0])      

    # def add_binary_col(df):
        # """
        # Functions to add binary column which tells if the data was missing or not
        # """
        # for label, content in df.items():
            # if pd.isnull(content).sum():
                # df["ismissing_"+label] = pd.isnull(content)
        # return df
    st.write(df)
    return df
  


