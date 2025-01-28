# Import libraries
import pandas as pd
import yfinance as yf
import datetime as dt
import numpy as np
from fredapi import Fred
import streamlit as st
import ssl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from dateutil.relativedelta import relativedelta
    #import library
from fredapi import Fred
import streamlit as st
from sklearn.pipeline import make_pipeline
from datetime import date

def stockDataFetcher(ticker):
    
    # #Set start and end date of data download
    start_date = "2014-09-01"
    end_date = "2024-08-31"
    
    # # Define function to get stock price history
    def get_stock_data(ticker):
         stock = yf.Ticker(ticker)
         df = stock.history(start=start_date, end=end_date, interval='1d')
         return df[['Open', 'Close', 'High', 'Low']]
    
    all_data = {ticker: get_stock_data(ticker)}
    
    # # Combine all data into a single DataFrame
    final_df_stock_prices = pd.concat(all_data, axis=1)
    final_df_stock_prices.columns = pd.MultiIndex.from_product([[ticker], ['Open', 'Close', 'High', 'Low']])
    return final_df_stock_prices




def getMacroData():

    ssl._create_default_https_context = ssl._create_unverified_context
    
    fred = Fred(api_key=st.secrets["API_KEY"])
    
    # Get historical GDP (Gross Domestic Product) data
    gdp = fred.get_series('GDP', observation_start='2014-09-01')
    
    # Get historical inflation data (CPI - Consumer Price Index)
    inflation = fred.get_series('CPIAUCSL', observation_start='2014-09-01')
    
    # Get historical unemployment rate data
    unemployment = fred.get_series('UNRATE', observation_start='2014-09-01')
    
    # Get historical data for Retail Sales
    retail_sales = fred.get_series('RSAFSNA', observation_start='2014-09-01')
    
    # Get historical data for Industrial Production (Index)
    industrial_production = fred.get_series('INDPRO', observation_start='2014-09-01')
    
    # Fetch Monthly Real GDP Index
    gdp_data = fred.get_series('A191RL1Q225SBEA', observation_start='2014-06-30')
    
    # Convert to DataFrame
    gdp_df = pd.DataFrame(gdp_data, columns=['Real GDP Index'])
    
    # Calculate Monthly GDP Growth (approximation)
    gdp_df['GDP Growth (%)'] = gdp_df['Real GDP Index'].pct_change() * 100
    
    
    # Create a DataFrame to store the data
    macro_economic_data = pd.DataFrame({
        'GDP Growth': gdp_df['GDP Growth (%)'],
        'Inflation (CPI)': inflation,
        'Unemployment Rate': unemployment,
        'Retail Sales': retail_sales,
        'Industrial Production': industrial_production
    })
    
    #Forward fill missing data
    macro_economic_data = macro_economic_data.fillna(method='ffill')
    
    # Display the first few rows of the data
    return macro_economic_data


def getEPS(ticker):
    

    url_eps = '../data/' + ticker + '_EPS.html'
    
    # Use pandas to read tables on the webpages
    tables_eps = pd.read_html(url_eps)
    
    # Select the first table from the list (index 0)
    df_eps = tables_eps[1]
    
    #Drop extra column index level, set index as date and drop extra columns
    df_eps = df_eps.rename(columns={df_eps.columns[0]: 'Date', df_eps.columns[1]: 'Quarterly EPS'})
    df_eps.set_index(df_eps.columns[0], inplace=True)
    df_eps.index = pd.to_datetime(df_eps.index)
    return df_eps


def getPE(ticker):

    url_pe = '../data/' + ticker + '_PE_Ratio.html'
    
    # Use pandas to read tables on the webpages
    table_pe = pd.read_html(url_pe)
    
    # Select the first table from the list (index 0)
    df_pe = table_pe[0]
    
    #Drop extra column index level, set index as date and drop extra columns
    df_pe = df_pe.droplevel(0, axis =1)
    df_pe.set_index(df_pe.columns[0], inplace=True)
    df_pe.drop(axis=1, columns=["Stock Price", "TTM Net EPS"], inplace=True)
    df_pe.index = pd.to_datetime(df_pe.index)
    return df_pe


def preProcessing(final_df_stock_prices, macro_economic_data, df_pe, df_eps):

    # Get rid of unneeded rows
    economic_data_needed = macro_economic_data.iloc[2:,:]

    #Convert to daily data from monthly data
    economic_data_needed.index = economic_data_needed.index.to_period('M')
    economic_data_needed = economic_data_needed.resample('D').ffill()

    # Converge indexes before merging
    final_df_stock_prices.reset_index(inplace=True)
    final_df_stock_prices['Date'] = final_df_stock_prices['Date'].astype("string").str[0:10]
    final_df_stock_prices = final_df_stock_prices.set_index('Date', drop=True)
    
    final_df_stock_prices.index = final_df_stock_prices.index.astype('str')
    economic_data_needed.index = economic_data_needed.index.astype('str')
    
    # Add multilevel index just as in stock price data before merging
    economic_data_needed.columns = pd.MultiIndex.from_product([['Macroeconomic'], economic_data_needed.columns])
    
    #Merge dataframes
    economic_data_needed = economic_data_needed.merge(final_df_stock_prices,how="left", left_index=True, right_index=True)
    
    economic_data_needed.index = pd.to_datetime(economic_data_needed.index)
    
    final_stocks_df = economic_data_needed.copy()
    final_stocks_df = final_stocks_df.dropna()
    
    final_df_nvidia = final_stocks_df.copy()
    final_df_nvidia_close = final_df_nvidia.drop(['Open', 'High', 'Low'], axis=1, level=1)
    # Drop extra level in the columns
    final_df_nvidia_close.columns = final_df_nvidia_close.columns.droplevel()
    final_df_nvidia_close = final_df_nvidia_close.rename(columns={'Close': 'NVIDIA_Close'})

    final_df_nvidia_close = final_df_nvidia_close.merge(df_pe, how='left', left_index = True, right_index = True)
    final_df_nvidia_close = final_df_nvidia_close.merge(df_eps, how='left', left_index = True, right_index = True)
    
    if(ticker=='NVDA'):
        final_df_nvidia_close['PE Ratio'].iloc[0] = df_pe.loc['2014-07-31']
        final_df_nvidia_close['Quarterly EPS'].iloc[0] = df_eps.loc['2014-07-31', 'Quarterly EPS']
    
    else:
        final_df_nvidia_close['PE Ratio'].iloc[0] = df_pe.loc['2014-06-30']
        final_df_nvidia_close['Quarterly EPS'].iloc[0] = df_eps.loc['2014-06-30', 'Quarterly EPS']
        
    final_df_nvidia_close['PE Ratio'] = final_df_nvidia_close['PE Ratio'].fillna(method="ffill")
    
    
    final_df_nvidia_close['Quarterly EPS'] = final_df_nvidia_close['Quarterly EPS'].fillna(method="ffill")
    
    final_df_nvidia_close['Quarterly EPS'] = final_df_nvidia_close['Quarterly EPS'].str.replace('$', '')

    final_df_nvidia_close['Rolling Mean'] = final_df_nvidia_close['NVIDIA_Close'].rolling(window=30).mean()
    final_df_nvidia_close['Rolling Std'] = final_df_nvidia_close['NVIDIA_Close'].rolling(window=30).std()
    
    # Convert "NVDIA_Close" to "float" from "string" for visualizations
    final_df_nvidia_close['NVIDIA_Close'] = final_df_nvidia_close['NVIDIA_Close'].astype(np.dtype("float32"))
    
    
    final_df_nvidia_close['Lagged_GDP'] = final_df_nvidia_close['GDP Growth'].shift(21)
    final_df_nvidia_close['Lagged_Inflation'] = final_df_nvidia_close['Inflation (CPI)'].shift(21)
    final_df_nvidia_close['Lagged_Unemp_rate'] = final_df_nvidia_close['Unemployment Rate'].shift(21)
    final_df_nvidia_close['Lagged_Retail_Sales'] = final_df_nvidia_close['Retail Sales'].shift(21)
    final_df_nvidia_close['Lagged_Industrial_prod'] = final_df_nvidia_close['Industrial Production'].shift(21)
    final_df_nvidia_close['Lagged_PE'] = final_df_nvidia_close['PE Ratio'].shift(21)
    final_df_nvidia_close['Lagged_EPS'] = final_df_nvidia_close['Quarterly EPS'].shift(21)
    
    final_df_nvidia_close['Lagged_Rolling_Mean'] = final_df_nvidia_close['Rolling Mean'].shift(21)
    final_df_nvidia_close['Lagged_Rolling_Std'] = final_df_nvidia_close['Rolling Std'].shift(21)
    
    final_df_nvidia_close = final_df_nvidia_close.dropna()
    return final_df_nvidia_close




def drawVisualizations(final_df_nvidia_close):

    def plot_reg(col):
        sns.lmplot(x=col, y='NVIDIA_Close', data=final_df_nvidia_close)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        # plt.savefig(f'pairs{col}.png', bbox_inches='tight')
    
    cols = [i for i in final_df_nvidia_close.columns.values.tolist() if i not in ["NVIDIA_Close", "Quarterly EPS", "Lagged_EPS"]]
    for col in cols:
        plot_reg(col)

    # Plot heatmap
    plt.figure(figsize=(20,20))
    sns.heatmap(final_df_nvidia_close.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

    # Plot NVIDIA Closing Price with Rolling Mean and Rolling Std Dev.

    final_df_nvidia_close['Rolling Mean'] = final_df_nvidia_close['NVIDIA_Close'].rolling(window=30).mean()
    final_df_nvidia_close['Rolling Std'] = final_df_nvidia_close['NVIDIA_Close'].rolling(window=30).std()
    final_df_nvidia_close[['NVIDIA_Close', 'Rolling Mean', 'Rolling Std']].plot()
    plt.xticks(rotation=90)
    #plt.savefig('ma.png', bbox_inches='tight') 
    # plt.show()
    st.pyplot(plt)



def trainAndTestLinearReg(final_df_nvidia_close):

    cols = ['Lagged_Rolling_Mean', 'Lagged_Unemp_rate', 'Lagged_GDP', 'Lagged_Inflation', 'Lagged_Industrial_prod', 'Lagged_Retail_Sales', 'Lagged_PE', 'Lagged_EPS']
    
    
    train_size = int(len(final_df_nvidia_close) * 0.8)  # 80% for training
    X_train = final_df_nvidia_close[:train_size][cols]
    y_train = final_df_nvidia_close[:train_size]['NVIDIA_Close']
    
    X_test = final_df_nvidia_close[train_size:][cols]
    y_test = final_df_nvidia_close[train_size:]['NVIDIA_Close']
    
    
    model = LinearRegression()
    
    

    cols = ['Lagged_Rolling_Mean', 'Lagged_Unemp_rate', 'Lagged_Inflation', 'Lagged_PE', 'Lagged_EPS']
    
    X = final_df_nvidia_close[cols]
    y = final_df_nvidia_close['NVIDIA_Close']
    
    
    train_size = int(len(final_df_nvidia_close) * 0.8)  # 80% for training
    X_train = final_df_nvidia_close[:train_size][cols]
    y_train = final_df_nvidia_close[:train_size]['NVIDIA_Close']
    
    X_test = final_df_nvidia_close[train_size:][cols]
    y_test = final_df_nvidia_close[train_size:]['NVIDIA_Close']
    
    # test = final_df_nvidia_close[train_size:]
    
    linreg = LinearRegression().fit(X_train, y_train)

    #st.write("The prediction is " + str(y_pred_baseline[0]))

    
    st.write('linear model coeff (w): {}'
         .format(linreg.coef_))
    st.write('linear model intercept (b): {:.3f}'
         .format(linreg.intercept_))
    st.write('R-squared score (training): {:.3f}'
         .format(linreg.score(X_train, y_train)))
    st.write('R-squared score (test): {:.3f}'
         .format(linreg.score(X_test, y_test)))
    
    y_pred = linreg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    st.write("R-sq is" + str(r2))
    
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Mean Absolute Error (MAE): {mae}")
    
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error (MSE): {mse}")
    
    # rmse = root_mean_squared_error(y_test, y_pred)
    # print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    mape = mean_absolute_percentage_error(y_test, y_pred)
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape}")
    
    y_test = y_test.to_numpy()
    
    
    # Calculate directional accuracy
    actual_directions = np.sign(y_test[1:] - y_test[:-1])
    predicted_directions = np.sign(y_pred[1:] - y_pred[:-1])
    
    directional_accuracy = np.mean(actual_directions == predicted_directions)
    
    st.write(f"Directional Accuracy: {directional_accuracy * 100:.2f}%")


    return linreg

def scaleData(final_df_nvidia_close):
    cols = ['Lagged_Rolling_Mean', 'Lagged_Unemp_rate', 'Lagged_GDP', 'Lagged_Inflation', 'Lagged_Industrial_prod', 'Lagged_Retail_Sales', 'Lagged_PE', 'Lagged_EPS']
    train_size = int(len(final_df_nvidia_close) * 0.8)  # 80% for training
    X_train = final_df_nvidia_close[:train_size][cols]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return scaler

def trainAndTestNeuralNetwork(final_df_nvidia_close):

    cols = ['Lagged_Rolling_Mean', 'Lagged_Unemp_rate', 'Lagged_GDP', 'Lagged_Inflation', 'Lagged_Industrial_prod', 'Lagged_Retail_Sales', 'Lagged_PE', 'Lagged_EPS']
    
    
    train_size = int(len(final_df_nvidia_close) * 0.8)  # 80% for training
    X_train = final_df_nvidia_close[:train_size][cols]
    y_train = final_df_nvidia_close[:train_size]['NVIDIA_Close']
    
    X_test = final_df_nvidia_close[train_size:][cols]
    y_test = final_df_nvidia_close[train_size:]['NVIDIA_Close']

    # Step 2: Scale the data
    X_train_scaled = scaleData(final_df_nvidia_close).transform(X_train)
    X_test_scaled = scaleData(final_df_nvidia_close).transform(X_test)
    
    # Define the Neural Network Regressor with L2 regularization (alpha parameter)
    mlp_regressor = MLPRegressor(max_iter=2000, learning_rate_init = 0.01, random_state=42)
    
    # Define the hyperparameter grid for GridSearchCV
    param_grid = {
        'hidden_layer_sizes': [(3,), (4,), (5,)],  # Number of neurons in each hidden layer
        'activation': ['relu', 'tanh', 'logistic'],    # Activation function
        'solver': ['adam', 'lbfgs', 'sgd'],                   # Optimization algorithm
        'alpha': [1e-7, 1e-6, 1e-5],                # L2 regularization term (higher = more regularization)
        'learning_rate': ['constant', 'adaptive'],     # Learning rate schedule
    }
    
    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=mlp_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    
    # Fit the model with GridSearchCV
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best hyperparameters and the best model
    st.write(f"Best hyperparameters: {grid_search.best_params_}")
    best_mlp_model = grid_search.best_estimator_
    
    # Predict on the test set
    y_pred6= best_mlp_model.predict(X_test_scaled)
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred6)
    st.write(f"Mean Squared Error on Test Set: {mse:.4f}")
    
    r2 = r2_score(y_test, y_pred6)
    st.write("R-sq is" + str(r2))

    return best_mlp_model






def getPredictorFeatureVariablesAndRunModel(ticker, date_to_predict, df_eps, df_pe, linreg):

    fred = Fred(api_key=st.secrets["API_KEY"])

    variables_date = date_to_predict - relativedelta(months=1)
    rolling_start_date = variables_date - relativedelta(months=2)
    
    #Grab data from yfinance api
    
    # #Set start and end date of data download
    start_date = rolling_start_date.strftime("%Y-%m-%d")
    end_date = variables_date.strftime("%Y-%m-%d")
    
    
    
    # # Define function to get stock price history
    def get_stock_data(ticker):
         stock = yf.Ticker(ticker)
         df = stock.history(start=start_date, end=end_date, interval='1d')
         return df[['Close']]
    
    # # Get stock price history for all the stocks in all_data dataframe
    stock_data = get_stock_data(ticker)

    rolling_mean = stock_data['Close'].rolling(window=30).mean().iloc[-1]



    ssl._create_default_https_context = ssl._create_unverified_context

    # Get historical GDP (Gross Domestic Product) data
    gdp_lagged = fred.get_series('GDP', observation_start=variables_date, observation_end=variables_date)
    
    # Get historical inflation data (CPI - Consumer Price Index)
    inflation_lagged = fred.get_series('CPIAUCSL', observation_start=variables_date, observation_end=variables_date)
    
    # Get historical unemployment rate data
    unemployment_lagged = fred.get_series('UNRATE', observation_start=variables_date, observation_end=variables_date)
    
    # Get historical data for Retail Sales
    retail_sales_lagged = fred.get_series('RSAFSNA', observation_start=variables_date, observation_end=variables_date)
    
    # Get historical data for Industrial Production (Index)
    industrial_production_lagged = fred.get_series('INDPRO', observation_start=variables_date, observation_end=variables_date)

    eps_lagged = float(df_eps.iloc[0]['Quarterly EPS'].replace("$", ""))
    pe_lagged = df_pe.iloc[0]['PE Ratio']
    

    cols = ['Lagged_Rolling_Mean', 'Lagged_Unemp_rate', 'Lagged_Inflation', 'Lagged_PE', 'Lagged_EPS']
    df_final = pd.DataFrame([[rolling_mean, unemployment_lagged, inflation_lagged, pe_lagged, eps_lagged]], columns=cols)

    X_pred = df_final[cols][:]
    

    y_pred_baseline = linreg.predict(X_pred)

    st.write("The prediction is " + str(y_pred_baseline[0]))


def getPredictorFeatureVariablesAndRunNeuralNetworkModel(ticker, date_to_predict, df_eps, df_pe, best_mlp_model, scaler):

    fred = Fred(api_key=st.secrets["API_KEY"])

    variables_date = date_to_predict - relativedelta(months=1)
    rolling_start_date = variables_date - relativedelta(months=2)
    
    #Grab data from yfinance api
    
    # #Set start and end date of data download
    start_date = rolling_start_date.strftime("%Y-%m-%d")
    end_date = variables_date.strftime("%Y-%m-%d")
    
    
    
    # # Define function to get stock price history
    def get_stock_data(ticker):
         stock = yf.Ticker(ticker)
         df = stock.history(start=start_date, end=end_date, interval='1d')
         return df[['Close']]
    
    # # Get stock price history for all the stocks in all_data dataframe
    stock_data = get_stock_data(ticker)

    rolling_mean = stock_data['Close'].rolling(window=30).mean().iloc[-1]



    ssl._create_default_https_context = ssl._create_unverified_context

    # Get historical GDP (Gross Domestic Product) data
    gdp_lagged = fred.get_series('GDP', observation_start=variables_date, observation_end=variables_date)
    
    # Get historical inflation data (CPI - Consumer Price Index)
    inflation_lagged = fred.get_series('CPIAUCSL', observation_start=variables_date, observation_end=variables_date)
    
    # Get historical unemployment rate data
    unemployment_lagged = fred.get_series('UNRATE', observation_start=variables_date, observation_end=variables_date)
    
    # Get historical data for Retail Sales
    retail_sales_lagged = fred.get_series('RSAFSNA', observation_start=variables_date, observation_end=variables_date)
    
    # Get historical data for Industrial Production (Index)
    industrial_production_lagged = fred.get_series('INDPRO', observation_start=variables_date, observation_end=variables_date)

    # Fetch Monthly Real GDP Index
    gdp_data = fred.get_series('A191RL1Q225SBEA', observation_start=variables_date, observation_end=variables_date)
    
    # Convert to DataFrame
    gdp_df = pd.DataFrame(gdp_data, columns=['Real GDP Index'])
    
    # Calculate Monthly GDP Growth (approximation)
    gdp_lagged = gdp_df['Real GDP Index'].pct_change() * 100

    eps_lagged = float(df_eps.iloc[0]['Quarterly EPS'].replace("$", ""))
    pe_lagged = df_pe.iloc[0]['PE Ratio']
    
    cols = ['Lagged_Rolling_Mean', 'Lagged_Unemp_rate', 'Lagged_GDP', 'Lagged_Inflation', 'Lagged_Industrial_prod', 'Lagged_Retail_Sales', 'Lagged_PE', 'Lagged_EPS']

    #cols = ['Lagged_Rolling_Mean', 'Lagged_Unemp_rate', 'Lagged_Inflation', 'Lagged_PE', 'Lagged_EPS']
    df_final = pd.DataFrame([[rolling_mean, unemployment_lagged, gdp_lagged, inflation_lagged, industrial_production_lagged, retail_sales_lagged, pe_lagged, eps_lagged]], columns=cols)

    X_pred = df_final[cols][:]

    X_pred_scaled = scaler.transform(X_pred)

    y_pred_date_req= best_mlp_model.predict(X_pred_scaled)

    st.write(f"The prediction is " + str(y_pred_date_req[0]))



def create_pipeline(ticker, dateToPredict):
    return make_pipeline(
        stockDataFetcher(ticker),
        getMacroData(),
        getEPS(ticker),
        getPE(ticker),
        preProcessing(stockDataFetcher(ticker), getMacroData(), getPE(ticker), getEPS(ticker)),
        drawVisualizations(preProcessing(stockDataFetcher(ticker), getMacroData(), getPE(ticker), getEPS(ticker))),
        trainAndTestLinearReg(preProcessing(stockDataFetcher(ticker), getMacroData(), getPE(ticker), getEPS(ticker))),
        #trainAndTestNeuralNetwork(preProcessing(stockDataFetcher(ticker), getMacroData(), getPE(ticker), getEPS(ticker))),
        getPredictorFeatureVariablesAndRunModel(ticker, dateToPredict, getEPS(ticker), getPE(ticker),trainAndTestLinearReg(preProcessing(stockDataFetcher(ticker), getMacroData(), getPE(ticker), getEPS(ticker))))
        )

def create_pipeline_advanced(ticker, dateToPredict):
    return make_pipeline(
        stockDataFetcher(ticker),
        getMacroData(),
        getEPS(ticker),
        getPE(ticker),
        scaleData(preProcessing(stockDataFetcher(ticker), getMacroData(), getPE(ticker), getEPS(ticker))),
        preProcessing(stockDataFetcher(ticker), getMacroData(), getPE(ticker), getEPS(ticker)),
        drawVisualizations(preProcessing(stockDataFetcher(ticker), getMacroData(), getPE(ticker), getEPS(ticker))),
        #trainAndTestLinearReg(preProcessing(stockDataFetcher(ticker), getMacroData(), getPE(ticker), getEPS(ticker))),
        trainAndTestNeuralNetwork(preProcessing(stockDataFetcher(ticker), getMacroData(), getPE(ticker), getEPS(ticker))),
        getPredictorFeatureVariablesAndRunNeuralNetworkModel(ticker, dateToPredict, getEPS(ticker), getPE(ticker),trainAndTestNeuralNetwork(preProcessing(stockDataFetcher(ticker), getMacroData(), getPE(ticker), getEPS(ticker))), scaleData(preProcessing(stockDataFetcher(ticker), getMacroData(), getPE(ticker), getEPS(ticker))))
        )


# App title
st.title("ML App: Predict Outcomes")

# User input
st.write("Provide input features:")
tickers = ['MSFT', 'AAPL', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA']
ticker = st.selectbox("Select a ticker:", tickers)
dateToPredict = st.date_input("Date (Please enter a date within one month from now.)")

# # Replace or add a dynamic step using FunctionTransformer
# dynamic_transformer = ('dynamic_transformer', FunctionTransformer(create_pipeline, kw_args={'ticker': ticker, 'dateToPredict': date}))
# pipeline.steps.insert(0, dynamic_transformer)  # Insert after the scaler


# Prediction
if st.button("Predict Using Liner Regression (Baseline Model)"):
    # prediction = pipeline.named_steps['create_pipeline'].ticker = ticker
    # prediction = pipeline.named_steps['create_pipeline'].date = date
    create_pipeline(ticker=ticker, dateToPredict=dateToPredict)
    #pipeline.cp()


if st.button("Predict Using Neural Network (Advanced ML Model)"):
    # prediction = pipeline.named_steps['create_pipeline'].ticker = ticker
    # prediction = pipeline.named_steps['create_pipeline'].date = date
    create_pipeline_advanced(ticker=ticker, dateToPredict=dateToPredict)
    #pipeline.cp()