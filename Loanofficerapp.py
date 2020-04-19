# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 07:10:38 2020

@author: willem van der Schans
"""


import time
import random    
random.seed(123)

import pandas as pd #Data Processing
import numpy as np # Analytics
np.random.seed(seed=123)
import matplotlib.pyplot as plt #Plots
import seaborn as sns #Plots
import warnings
from scipy import stats
from scipy.stats import norm, skew
# Set seaborn settings
color = sns.color_palette()
#sns.set.style('darkgrid')

def ingore_warn(*args, **kwargs):
    pass
warnings.warn = ingore_warn #Ignore warnings
    

# ### Create Applet

# Lists to show in Combos

train = pd.read_excel("https://github.com/Kydoimos97/FINAN6500DATA/raw/master/data/SBA_training_data.xlsx")
recession = pd.read_csv("https://github.com/Kydoimos97/FINAN6500DATA/raw/master/data/JHDUSRGDPBR.csv")


# Recession Code 
recession.rename(columns = {"JHDUSRGDPBR" : "Recession", "DATE" : "Date"}, inplace=True)

recession['Date'] = pd.to_datetime(recession['Date'])
recession['Recession'] = recession['Recession'].astype('int64').astype(dtype="category")

recession['Year'] = recession['Date'].dt.year

recession = recession.reset_index()
recession = recession.pivot(index='index', columns='Year', values='Recession')
recession = recession.mode(axis = 0, dropna=True)
recession = recession.drop(index=recession.index.difference([0]))

keys = tuple(recession.columns)
values = tuple(recession.to_numpy()[0])

rename_dict = dict(zip(keys, values))

train['Recession'] = train['ApprovalFY']
train['Recession'] = train['Recession'].replace(rename_dict)
train['Recession'].fillna(0, inplace=True)

#Number 3  

for i in train.index:
    if train.iloc[i,train.columns.get_loc('RevLineCr')] != "Y":
        train.at[i, 'RevLineCr'] = "N"
    else:
        train.at[i, 'RevLineCr'] = "Y"
    



for i in train.index:
    if train.iloc[i,train.columns.get_loc('LowDoc')] != "Y":
        train.at[i, 'LowDoc'] = "N"
    else:
        train.at[i, 'LowDoc'] = "Y"
        



#Number 5
train['ChgOffDate'] = train['ChgOffDate'].fillna(0)

for i in train.index:
    if train.iloc[i,train.columns.get_loc('ChgOffDate')] != 0:
        train.at[i, 'ChgOffDate'] = 1
    else:
        train.at[i, 'ChgOffDate'] = 0
        



#Number 10 


for i in train.index:
    if train.iloc[i,train.columns.get_loc('FranchiseCode')] > 1:
        train.at[i, 'FranchiseCode'] = 1
    else:
        train.at[i, 'FranchiseCode'] = 0
        


#Number 1
train.drop(columns=['LoanNr_ChkDgt', 'Name'], inplace=True)

# Number 2
train.drop(columns=['NAICS'], inplace=True)

# Number 6
train.dropna(subset=['Bank'], inplace=True)
train.dropna(subset=['BankState'], inplace=True)

# Number 7
train.drop(columns=['BalanceGross'], inplace=True)

# Number 8
train.dropna(subset=['MIS_Status'], inplace=True)

#Number 9
train.drop(columns=['City'], inplace=True)
train.drop(columns=['Zip'], inplace=True)

#Remove Specific Dates?
train.drop(columns=['ApprovalDate'], inplace=True)
train.drop(columns=['DisbursementDate'], inplace=True)




train.to_csv("train.csv",index=False)
z = train

#Picture
import urllib.request
urllib.request.urlretrieve("https://github.com/Kydoimos97/FINAN6500DATA/raw/master/data/logo.gif", "logo.gif")

## States
states = sorted(list(z['BankState']))
states = list( dict.fromkeys(states) )

## Banks
banks = sorted(list(z['Bank']))
banks = list( dict.fromkeys(banks) )

## Binary Choices


### Submit Functions
#def check_number_NAICS():
    #value = NAICS.get()
    #if value.isdigit():
        #NAICS_text_box.delete(1.0, "end-1c")
        #NAICS_text_box.insert("end-1c","Submission Accepted")
    #else:
        #NAICS_text_box.delete(1.0, "end-1c")
        #NAICS_text_box.insert("end-1c","Incorrect Submission")
    
def check_number_FY():
    value = fiscal_year.get()
    if value.isdigit():
        if len(value) == 4:
            FY_text_box.delete(1.0, "end-1c")
            FY_text_box.insert("end-1c","Submission Accepted")
        else: 
            FY_text_box.delete(1.0, "end-1c")
            FY_text_box.insert("end-1c","Incorrect Submission")   
    else:
        FY_text_box.delete(1.0, "end-1c")
        FY_text_box.insert("end-1c","Incorrect Submission")        
        
def check_number_Term():
    value = Term.get()
    if value.isdigit():
        Term_text_box.delete(1.0, "end-1c")
        Term_text_box.insert("end-1c","Submission Accepted")
    else:
        Term_text_box.delete(1.0, "end-1c")
        Term_text_box.insert("end-1c","Incorrect Submission")      
        
def check_number_NoEmp():
    value = NoEmp.get()
    if value.isdigit():
        NoEmp_text_box.delete(1.0, "end-1c")
        NoEmp_text_box.insert("end-1c","Submission Accepted")
    else:
        NoEmp_text_box.delete(1.0, "end-1c")
        NoEmp_text_box.insert("end-1c","Incorrect Submission")    
                
def check_number_CreateJob():
    value = CreateJob.get()
    if value.isdigit():
        CreateJob_text_box.delete(1.0, "end-1c")
        CreateJob_text_box.insert("end-1c","Submission Accepted")
    else:
        CreateJob_text_box.delete(1.0, "end-1c")
        CreateJob_text_box.insert("end-1c","Incorrect Submission") 

def check_number_RetainJob():
    value = RetainJob.get()
    if value.isdigit():
        RetainJob_text_box.delete(1.0, "end-1c")
        RetainJob_text_box.insert("end-1c","Submission Accepted")
    else:
        RetainJob_text_box.delete(1.0, "end-1c")
        RetainJob_text_box.insert("end-1c","Incorrect Submission") 


def check_number_DisbursementGross ():
    value = DisbursementGross.get()
    if value.isdigit():
        DisbursementGross_text_box.delete(1.0, "end-1c")
        DisbursementGross_text_box.insert("end-1c","Submission Accepted")
    else:
        DisbursementGross_text_box.delete(1.0, "end-1c")
        DisbursementGross_text_box.insert("end-1c","Incorrect Submission")         
     
def check_number_GrAppv ():
    value = GrAppv.get()
    if value.isdigit():
        GrAppv_text_box.delete(1.0, "end-1c")
        GrAppv_text_box.insert("end-1c","Submission Accepted")
    else:
        GrAppv_text_box.delete(1.0, "end-1c")
        GrAppv_text_box.insert("end-1c","Incorrect Submission")    
        
def check_number_SBA_Appv ():
    value = SBA_Appv.get()
    if value.isdigit():
        SBA_Appv_text_box.delete(1.0, "end-1c")
        SBA_Appv_text_box.insert("end-1c","Submission Accepted")
    else:
        SBA_Appv_text_box.delete(1.0, "end-1c")
        SBA_Appv_text_box.insert("end-1c","Incorrect Submission")   
        
        
def getInput():

    sub_bank_sub = Bank_Input.get()
    sub_bank_state = Bank_State_Input.get()
    sub_customer_state = State_Input.get()
    #sub_naics = NAICS.get()
    sub_fy = fiscal_year.get()
    sub_term = Term.get()
    sub_noemp = NoEmp.get()
    sub_newexist = BT_Input.get()
    sub_createjob = CreateJob.get()
    sub_retainedjob = RetainJob.get()
    sub_franchise = Franchise_Input.get()
    sub_urbanrural= UrbanRural_Input.get()
    sub_disimbursementgross = DisbursementGross.get()
    sub_grapprv = GrAppv.get()
    sub_sbaapprv = SBA_Appv.get()
    sub_revlinecr = RevLineCr_Input.get()
    sub_lowdoc = LowDoc_Input.get()
    sub_recession = Recession_Input.get()
    
    global loan_submission
    loan_submission = {'State' : [sub_customer_state],
                       'Bank': [sub_bank_sub], 
                       'BankState' : [sub_bank_state], 
                       #'NAICS' : [sub_naics], 
                       'ApprovalFY' : [sub_fy], 
                       'Term' : [sub_term], 
                       'NoEmp' : [sub_noemp], 
                       'NewExist' : [sub_newexist], 
                       'CreateJob' : [sub_createjob], 
                       'RetainedJob' : [sub_retainedjob], 
                       'FranchiseCode' : [sub_franchise], 
                       'UrbanRural' : [sub_urbanrural], 
                       'RevLineCr': [sub_revlinecr], 
                       'LowDoc' : [sub_lowdoc], 
                       'DisbursementGross' : [sub_disimbursementgross], 
                       'GrAppv' : [sub_grapprv], 
                       'SBA_Appv' : [sub_sbaapprv], 
                       'Recession' : [sub_recession]}
    
    loan_submission = pd.DataFrame(data=loan_submission)
    loan_submission.to_csv("last_loan_submission.csv",index=False)
    
    df_loan_submission = loan_submission

    # Same Transformations as we do to train
    
    #Number 3   
    for i in df_loan_submission.index:
        if df_loan_submission.iloc[i,df_loan_submission.columns.get_loc('RevLineCr')] != "Yes":
            df_loan_submission.at[i, 'RevLineCr'] = "N"
        else:
            df_loan_submission.at[i, 'RevLineCr'] = "Y"

    #Number 4  
    
    for i in df_loan_submission.index:
        if df_loan_submission.iloc[i,df_loan_submission.columns.get_loc('LowDoc')] != "Yes":
            df_loan_submission.at[i, 'LowDoc'] = "N"
        else:
            df_loan_submission.at[i, 'LowDoc'] = "Y"
   
    
    #Number 10     
    for i in df_loan_submission.index:
        if df_loan_submission.iloc[i,df_loan_submission.columns.get_loc('FranchiseCode')] == "Yes":
            df_loan_submission.at[i, 'FranchiseCode'] = 1
        else:
            df_loan_submission.at[i, 'FranchiseCode'] = 0
    
    # NewExist
    
    for i in df_loan_submission.index:
        if df_loan_submission.iloc[i,df_loan_submission.columns.get_loc('NewExist')] == "Existing":
            df_loan_submission.at[i, 'NewExist'] = 1
        else:
            df_loan_submission.at[i, 'NewExist'] = 2
            
    # Recession
    
    for i in df_loan_submission.index:
        if df_loan_submission.iloc[i,df_loan_submission.columns.get_loc('Recession')] == "Yes":
            df_loan_submission.at[i, 'Recession'] = 1
        else:
            df_loan_submission.at[i, 'Recession'] = 0
    
    # Urban Rural
    
    for i in df_loan_submission.index:
        if df_loan_submission.iloc[i,df_loan_submission.columns.get_loc('UrbanRural')] == "Urban":
            df_loan_submission.at[i, 'UrbanRural'] = 1
        elif df_loan_submission.iloc[i,df_loan_submission.columns.get_loc('UrbanRural')] == "Rural":
            df_loan_submission.at[i, 'UrbanRural'] = 2
        else: 
            df_loan_submission.at[i, 'UrbanRural'] = 0
    
    #Merge Data Sets
    z = pd.read_csv('train.csv')
    loan_train = z
    loan_test = df_loan_submission
    loan_train_shape = loan_train.shape[0]
    loan_test_shape = df_loan_submission.shape[0]
    
    #Save Target Variables
    loan_y_ChgOffDate = loan_train.ChgOffDate.values
    loan_y_ChgOffDate=loan_y_ChgOffDate.astype('int')
    
    #Drop Targets
    if loan_train.shape[1] != 19:
        loan_train.drop(['ChgOffDate', 'MIS_Status', 'ChgOffPrinGr'], axis=1, inplace=True)
    else:
        pass
    
    # Convert Data Types
    
    toint = list(loan_train.dtypes[loan_train .dtypes == "int64"].index)
    
    for i in toint:
        loan_test[i] = loan_test[i].astype(str).astype(int)
    
    
    # Merge Data sets.
    loan_alldata = pd.concat((loan_train, loan_test)).reset_index(drop=True)

    
    
    from sklearn.preprocessing import LabelEncoder
    
    lbl = LabelEncoder() 
    lbl.fit(list(loan_alldata['Bank'].values)) 
    loan_alldata['Bank'] = lbl.transform(list(loan_alldata['Bank'].values))
    
    numeric_feats = loan_alldata.dtypes[loan_alldata.dtypes == "int64"].index
    
    
    # Check the skew of all numerical features
    skewed_feats = loan_alldata[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness.head(21)
    
    
    
    skewness = skewness[abs(skewness) > .75]
    skewness.dropna(inplace=True)

    
    from scipy.special import boxcox1p
    skewed_features = skewness.index
    lam = 0.15 #Regularly Picked Value
    for feat in skewed_features:
        loan_alldata[feat] = boxcox1p(loan_alldata[feat], lam)
    
    
    numeric_feats = loan_alldata.dtypes[loan_alldata.dtypes == "int64"].index
    
    # Check the skew of all numerical features
    skewed_feats = loan_alldata[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness.head(21)
    
    
    
    loan_alldata = pd.get_dummies(loan_alldata)

    
    loan_train = loan_alldata[:loan_train_shape]
    loan_test = loan_alldata[loan_train_shape:]
    
    
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    
    #Random Forest
    from sklearn.ensemble import RandomForestClassifier
    
    # XGBoost
    import xgboost as xgb
    
    model_RFC = RandomForestClassifier(criterion = 'gini', max_features = 'auto', n_estimators = 290, max_depth=55)


    model_xgb = xgb.XGBClassifier(colsample_bytree=.6, gamma=.4, 
                             learning_rate=0.08, max_depth=8, 
                             min_child_weight=5, n_estimators=230,
                             reg_alpha=0, reg_lambda = 0.41,
                             subsample=.8, silent=1, scale_pos_weight = 1,
                             random_state =123, nthread = -1)
    
    loan_y_train = loan_y_ChgOffDate
    
    model_RFC.fit(loan_train, loan_y_train) #Train on Full Train Data
    loan_RFC_pred = model_RFC.predict(loan_test) #Predict on Test
    
        
    model_xgb.fit(loan_train, loan_y_train) #Train on Full Train Data
    loan_xgb_pred = model_xgb.predict(loan_test) #Predict on Test
    
    if loan_RFC_pred == 0:
        if loan_xgb_pred == 0:
            tk.messagebox.showinfo(title = 'Loan Outcome',message = 'No Default predicted')
        else:
            tk.messagebox.showwarning(title = 'Loan Outcome',message = 'Only XGradient Boost Predicts Default')
    elif loan_xgb_pred == 0:
        tk.messagebox.showwarning(title = 'Loan Outcome',message = 'Only Random Forest predicts Default')
    else:
        tk.messagebox.showerror(title = 'Loan Outcome',message = 'Both Alogrithms Predict Default')
    

        
import tkinter as tk
from tkinter import ttk
from tkinter import *
 
app = tk.Tk() 
app.geometry('600x560')
app.title('Loan officer applet')

s=ttk.Style()
s.theme_use('alt')


#Explanation
logo = tk.PhotoImage(file="logo.gif",master=app)

w1 = tk.Label(app, image=logo).grid(column=2, row=0, columnspan = 2)

explanation = """Fill in the Customer Information. Use the buttons for free input
to check if the data is correct. If one Algorithm predicts default, 
research to make a better-informed decision."""

w2 = tk.Label(app, 
              justify=tk.LEFT,
              padx = 0, 
              text=explanation).grid(column=0, row=0, columnspan = 2)

# Bank
label1 = tk.Label(app,text = "")
label1.grid(column=0, row=1)

label0 = tk.Label(app,text = "Choose Bank")
label0.grid(column=0, row=1)

Bank_Input = ttk.Combobox(app, values=banks)
Bank_Input.grid(column=1, row=1)
# Bank_State
label2 = tk.Label(app,text = "Choose Bank State")
label2.grid(column=0, row=2)

Bank_State_Input = ttk.Combobox(app, values=states)
Bank_State_Input.grid(column=1, row=2)

# State
label3 = tk.Label(app,text = "Choose Customer State")
label3.grid(column=0, row=3)

State_Input = ttk.Combobox(app, values=states)
State_Input.grid(column=1, row=3)

# NAICS
#tk.Label(app, text="Input NAICS number").grid(row=4)
#NAICS = tk.Entry(app)
#NAICS.grid(row=4, column=1)

#submit_btn1 = Button(app, text="Check", width=10, command=check_number_NAICS)
#submit_btn1.grid(row=4, column=2)

#NAICS_text_box = tk.Text(app, width = 20, height = 1)
#NAICS_text_box.grid(row = 4, column = 3)
#NAICS_text_box.insert("end-1c","waiting")

#Fiscal Year
tk.Label(app, text="Input Fiscal Year").grid(row=4)
fiscal_year = tk.Entry(app)
fiscal_year.grid(row=4, column=1)

submit_btn2 = Button(app, text="Check", width=10, command=check_number_FY)
submit_btn2.grid(row=4, column=2)

FY_text_box = tk.Text(app, width = 20, height = 1)
FY_text_box.grid(row = 4, column = 3)
FY_text_box.insert("end-1c","waiting")

#Loan Term
tk.Label(app, text="Loan Term in Months").grid(row=5)
Term = tk.Entry(app)
Term.grid(row=5, column=1)

submit_btn3 = Button(app, text="Check", width=10, command=check_number_Term)
submit_btn3.grid(row=5, column=2)

Term_text_box = tk.Text(app, width = 20, height = 1)
Term_text_box.grid(row = 5, column = 3)
Term_text_box.insert("end-1c","waiting")

# NoEmp Term
tk.Label(app, text="Number of Employees").grid(row=6)
NoEmp = tk.Entry(app)
NoEmp.grid(row=6, column=1)

submit_btn4 = Button(app, text="Check", width=10, command=check_number_NoEmp)
submit_btn4.grid(row=6, column=2)

NoEmp_text_box = tk.Text(app, width = 20, height = 1)
NoEmp_text_box.grid(row = 6, column = 3)
NoEmp_text_box.insert("end-1c","waiting")

## Existing
tk.Label(app, text="Business Type").grid(row=7, column = 0)
#Existing = tk.Radiobutton(app, text = "Existing Business", value= 1).grid(row=7, column=1)
#NotExisting = tk.Radiobutton(app, text = "New Business", value= 2).grid(row=7, column=2)

tk.Label(app,text = "Franchise")
label1.grid(column=0, row=7)

BT_Input = ttk.Combobox(app, values=["Existing", "New"])
BT_Input.grid(column=1, row=7)

# Number of Jobs Created
tk.Label(app, text="Number of Jobs Created").grid(row=8)
CreateJob = tk.Entry(app)
CreateJob.grid(row=8, column=1)

submit_btn5 = Button(app, text="Check", width=10, command=check_number_CreateJob)
submit_btn5.grid(row=8, column=2)

CreateJob_text_box = tk.Text(app, width = 20, height = 1)
CreateJob_text_box.grid(row =8, column = 3)
CreateJob_text_box.insert("end-1c","waiting")

# Number of Jobs Retained
tk.Label(app, text="Number of Jobs Retained").grid(row=9)
RetainJob = tk.Entry(app)
RetainJob.grid(row=9, column=1)

submit_btn6 = Button(app, text="Check", width=10, command=check_number_RetainJob)
submit_btn6.grid(row=9, column=2)

RetainJob_text_box = tk.Text(app, width = 20, height = 1)
RetainJob_text_box.grid(row = 9, column = 3)
RetainJob_text_box.insert("end-1c","waiting")

## Franchise
label1 = tk.Label(app,text = "Franchise")
label1.grid(column=0, row=10)

Franchise_Input = ttk.Combobox(app, values=["Yes", "No"])
Franchise_Input.grid(column=1, row=10)

## UrbanRural
label1 = tk.Label(app,text = "Urban or Rural")
label1.grid(column=0, row=11)

UrbanRural_Input = ttk.Combobox(app, values=["Urban", "Rural", "Undefined"])
UrbanRural_Input.grid(column=1, row=11)

## DisbursementGross
tk.Label(app, text="Disbursement Gross").grid(row=12)
DisbursementGross = tk.Entry(app)
DisbursementGross.grid(row=12, column=1)

submit_btn6 = Button(app, text="Check", width=10, command=check_number_DisbursementGross)
submit_btn6.grid(row=12, column=2)

DisbursementGross_text_box = tk.Text(app, width = 20, height = 1)
DisbursementGross_text_box.grid(row = 12, column = 3)
DisbursementGross_text_box.insert("end-1c","waiting")

## GrAppv
tk.Label(app, text="Loan Amount").grid(row=13)
GrAppv = tk.Entry(app)
GrAppv.grid(row=13, column=1)

submit_btn7 = Button(app, text="Check", width=10, command=check_number_GrAppv)
submit_btn7.grid(row=13, column=2)

GrAppv_text_box = tk.Text(app, width = 20, height = 1)
GrAppv_text_box.grid(row = 13, column = 3)
GrAppv_text_box.insert("end-1c","waiting")

## SBA_Appv
tk.Label(app, text="SBA's guaranteed amount").grid(row=14)
SBA_Appv = tk.Entry(app)
SBA_Appv.grid(row=14, column=1)

submit_btn8 = Button(app, text="Check", width=10, command=check_number_SBA_Appv)
submit_btn8.grid(row=14, column=2)

SBA_Appv_text_box = tk.Text(app, width = 20, height = 1)
SBA_Appv_text_box.grid(row = 14, column = 3)
SBA_Appv_text_box.insert("end-1c","waiting")

## RevLineCr
label1 = tk.Label(app,text = "Revolving Line of Credit")
label1.grid(column=0, row=15)

RevLineCr_Input = ttk.Combobox(app, values=["Yes", "No"])
RevLineCr_Input.grid(column=1, row=15)

## LowDoc
label1 = tk.Label(app,text = "Low Doc Loan")
label1.grid(column=0, row=16)

LowDoc_Input = ttk.Combobox(app, values=["Yes", "No"])
LowDoc_Input.grid(column=1, row=16)

## LowDoc
label1 = tk.Label(app,text = "Economic Recession")
label1.grid(column=0, row=17)

Recession_Input = ttk.Combobox(app, values=["Yes", "No"])
Recession_Input.grid(column=1, row=17)

## Submit
Submit_btn = Button(app, text="Submit and Run",command = getInput, height=2, width=20,  bg='#cc0000', fg='white')
Submit_btn.grid(row=21, column=1, columnspan = 2,padx=10, pady=25)

print(Bank_Input.current(), Bank_Input.get())

app.mainloop()


