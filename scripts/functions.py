import ipywidgets as widgets
from IPython.display import display
import requests
import pandas as pd
import sqlite3
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # Import EarlyStopping
import keras_tuner as kt
import sqlite3
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

import sys
sys.path.append('../keys')  

from config import ALPHA_VANTAGE_API_KEY

# Connect to DB and import data into a pandas dataframe
connection = sqlite3.connect('../data/db2.sqlite')
averages = pd.read_sql_query("SELECT * FROM averages", connection)
averagesJunk = pd.read_sql_query("SELECT * FROM averagesJunk", connection)
averagesInvestment = pd.read_sql_query("SELECT * FROM averagesInvestment", connection)

connection.close()

# Functions to call the API data
def getData(ticker, ALPHA_VANTAGE_API_KEY):
    # Pulling from two parts of the API
    # See Documentation Here: https://www.alphavantage.co/documentation/
    income_statement_url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}'
    balance_sheet_url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}'
    cash_flow_url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}'

    try:
        # Fetching from multiple sources:
        # Income statement
        income_statement_response = requests.get(income_statement_url)
        income_statement_data = income_statement_response.json()
        
        # Balance sheet
        balance_sheet_response = requests.get(balance_sheet_url)
        balance_sheet_data = balance_sheet_response.json()

        # Cash_flow
        cash_flow_response = requests.get(cash_flow_url)
        cash_flow_data = cash_flow_response.json()

        # Check if data is valid before creating DataFrames
        if 'annualReports' in income_statement_data and 'annualReports' in balance_sheet_data and 'annualReports' in cash_flow_data:
            income_statement_df = pd.DataFrame(income_statement_data['annualReports'])
            balance_sheet_df = pd.DataFrame(balance_sheet_data['annualReports'])
            cash_flow_df = pd.DataFrame(cash_flow_data['annualReports'])

            return income_statement_df, balance_sheet_df, cash_flow_df
        else:
            raise ValueError("Invalid data format")
    except (KeyError, IndexError, requests.exceptions.RequestException, ValueError) as e:
        print(f"Error fetching financial data: {e}")
        return None, None, None


# Process the dataframes that come out of the getData API call
def transform_dataframe(df):
    num_cols = len(df.columns) - 2
    cols = list(df.columns[2:])
    transformed_df = df[cols]
    transformed_df = transformed_df.apply(pd.to_numeric, errors='coerce', axis=1)
    return transformed_df


# Calculate metrics based on the dataframes that come out of the API call

def calculate_metrics(balance_sheet_df, income_statement_df, cash_flow_df):
    # Extracting needed variables
    totalCurrentAssets = balance_sheet_df['totalCurrentAssets'].iloc[0]
    totalCurrentLiabilities = balance_sheet_df['totalCurrentLiabilities'].iloc[0]
    currentRatio = totalCurrentAssets / totalCurrentLiabilities
    
    longTermDebt = balance_sheet_df['longTermDebt'].iloc[0]
    totalShareholderEquity = balance_sheet_df['totalShareholderEquity'].iloc[0]
    longTermDebtCapital = longTermDebt / (longTermDebt + totalShareholderEquity)
    
    totalLiabilities = balance_sheet_df['totalLiabilities'].iloc[0]
    debtEquityRatio = totalLiabilities / totalShareholderEquity
    
    totalRevenue = income_statement_df['totalRevenue'].iloc[0]
    costofGoodsAndServicesSold = income_statement_df['costofGoodsAndServicesSold'].iloc[0]
    grossMargin = ((totalRevenue - costofGoodsAndServicesSold) / totalRevenue) * 100
    
    operatingIncome = income_statement_df['operatingIncome'].iloc[0]
    operatingMargin = operatingIncome / totalRevenue
    
    operatingExpenses = income_statement_df['operatingExpenses'].iloc[0]
    ebitMargin = ((totalRevenue - costofGoodsAndServicesSold - operatingExpenses) / totalRevenue) * 100
    
    incomeBeforeTax = income_statement_df['incomeBeforeTax'].iloc[0]
    depreciationAndAmortization = income_statement_df['depreciationAndAmortization'].iloc[0]
    ebitdaMargin = (incomeBeforeTax + depreciationAndAmortization) / totalRevenue
    
    preTaxProfitMargin = (incomeBeforeTax / totalRevenue) * 100
    
    netIncome = income_statement_df['netIncome'].iloc[0]
    netProfitMargin = (netIncome / totalRevenue) * 100
    
    totalAssets = balance_sheet_df['totalAssets'].iloc[0]
    totalAssetsPrevious = balance_sheet_df['totalAssets'].iloc[1]
    assetTurnoverRatio = totalRevenue / ((totalAssets + totalAssetsPrevious) / 2)
    
    totalShareholderEquityPrevious = balance_sheet_df['totalShareholderEquity'].iloc[1]
    returnOnEquity = netIncome / ((totalShareholderEquity + totalShareholderEquityPrevious) / 2)
    
    avgShareholderEquity = (totalShareholderEquity + totalShareholderEquityPrevious) / 2
    intangibleAssets = balance_sheet_df['intangibleAssets'].iloc[0]
    returnOnTangibleEquity = netIncome / (avgShareholderEquity - intangibleAssets)
    
    returnOnAssets = netIncome / totalAssets
    
    returnOnInvestment = (netIncome / ((totalShareholderEquity + totalShareholderEquityPrevious) / 2)) * 100
    
    operatingCashflow = cash_flow_df['operatingCashflow'].iloc[0]
    commonStockSharesOutstanding = balance_sheet_df['commonStockSharesOutstanding'].iloc[0]
    operatingCashFlowPerShare = operatingCashflow / commonStockSharesOutstanding
    
    capitalExpenditures = cash_flow_df['capitalExpenditures'].iloc[0]
    freeCashFlowPerShare = (operatingCashflow - capitalExpenditures) / commonStockSharesOutstanding

    # Create DataFrame
    metrics_df = pd.DataFrame({
        "Current Ratio": [currentRatio],
        "Long-term Debt / Capital": [longTermDebtCapital],
        "Debt/Equity Ratio": [debtEquityRatio],
        "Gross Margin": [grossMargin],
        "Operating Margin": [operatingMargin],
        "EBIT Margin": [ebitMargin],
        "EBITDA Margin": [ebitdaMargin],
        "Pre-Tax Profit Margin": [preTaxProfitMargin],
        "Net Profit Margin": [netProfitMargin],
        "Asset Turnover": [assetTurnoverRatio],
        "ROE - Return On Equity": [returnOnEquity],
        "Return On Tangible Equity": [returnOnTangibleEquity],
        "ROA - Return On Assets": [returnOnAssets],
        "ROI - Return On Investment": [returnOnInvestment],
        "Operating Cash Flow Per Share": [operatingCashFlowPerShare],
        "Free Cash Flow Per Share": [freeCashFlowPerShare]
    })

    for column in metrics_df.columns:
        # Step 2: Check for NaN values in each column
        nan_indices = metrics_df[column].isnull()
    
        # Step 3: Replace NaN values with corresponding averages
        metrics_df.loc[nan_indices, column] = averages.loc[0, column]
    
    return metrics_df


# Function to load the models and make predictions
def load_model(model_type, model_number, data_source):
    # Load the model
    loaded_model = joblib.load(f'../models/{model_type}/model{model_number}.joblib')
    data_frame = data_source

    # Define features set for new data
    X_new = data_frame.copy()

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Fit the StandardScaler
    X_scaler = scaler.fit(X_new)

    # Scale the new data
    X_new_scaled = X_scaler.transform(X_new)

    # Make predictions using the loaded model
    predictions_new = loaded_model.predict(X_new_scaled)
    
    return predictions_new

# Master function to process data and make predictions
# Master function to process data and make predictions
def process_data_and_predict(ticker, model_type, model_number):
    # Step 1: Get financial data
    income_statement_df, balance_sheet_df, cash_flow_df = getData(ticker, ALPHA_VANTAGE_API_KEY)
    
    # Check if any of the dataframes are None
    if income_statement_df is None or balance_sheet_df is None or cash_flow_df is None:
        print("Error: Unable to fetch financial data.")
        return None, None
    
    # Step 2: Transform data
    transformed_balance_sheet_df = transform_dataframe(balance_sheet_df)
    transformed_income_statement_df = transform_dataframe(income_statement_df)
    transformed_cash_flow_df = transform_dataframe(cash_flow_df)
    
    # Step 3: Calculate metrics
    metrics_df = calculate_metrics(transformed_balance_sheet_df, transformed_income_statement_df, transformed_cash_flow_df)
    
    # Step 4: Load model and make predictions
    predictions = load_model(model_type, model_number, metrics_df)
    
    return predictions, metrics_df

