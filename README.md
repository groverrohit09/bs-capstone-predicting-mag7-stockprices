**📈 Predicting Stock Prices of Mag-7 Stocks**<br>
A data science project aimed at forecasting the long-term performance of the Magnificent Seven (Mag-7) stocks over a year using fundamental and technical analysis.<br><br>

**📄 Executive Summary**<br>
The Magnificent Seven (Mag-7) stocks—Alphabet, Amazon, Apple, Meta Platforms, Microsoft, NVIDIA, and Tesla—collectively hold a market capitalization of approximately $14 trillion. These tech giants are pivotal in high-growth sectors like cloud computing and artificial intelligence. This project leverages data science techniques to predict their stock performance over the next year.<br>
The results can be used by investors to optimize their portfolio.<br><br>



**📊 Data Collection**<br>
Data was sourced from:<br>

Fundamental Data: Financial statements and key metrics.<br>These included -<br>
1. US GDP Growth<br>
2. Inflation (CPI) reported
3. Unemployment Rate reporrted
4. Reatil Sales number
5. Industrial Production number
6. PE Ration last reported.
7. Quarterly EPS last reported.<br>

Technical Data: Historical stock prices<br> The feature used was -<br>
1. 30 days Rolling Mean<br>




**Data sources include:**<br>



Yahoo Finance<br>
Alpha Vantage<br><br><br>



**🛠️ Methodology**<br>


Data Preprocessing: Cleaning and organizing data for analysis.<br>

Feature Engineering: Creating relevant features from raw data.<br>

Model Selection: Evaluating various machine learning models.<br>

Training & Validation: Training models and validating performance.<br>

Prediction: Forecasting stock prices for the upcoming year.<br><br>


**EDA**

<img width="391" alt="image" src="https://github.com/user-attachments/assets/f7a2f237-c5ea-47c0-a937-90a07d0458c9" />        <img width="398" alt="image" src="https://github.com/user-attachments/assets/c5353668-5a00-4fb1-982d-13eeda025b9d" /><br>
<img width="313" alt="image" src="https://github.com/user-attachments/assets/07ea21d5-56d7-44f0-96c1-3dbff3d49a02" />        <img width="245" alt="image" src="https://github.com/user-attachments/assets/230e652d-3417-42be-8256-4ffc8592cdd5" />        <img width="313" alt="image" src="https://github.com/user-attachments/assets/73dafc6f-7856-4a14-b431-b16d0f6032f4" /><br>
<img width="313" alt="image" src="https://github.com/user-attachments/assets/c424127a-7ae0-4cfb-8b60-d0034b28b12a" />        <img width="304" alt="image" src="https://github.com/user-attachments/assets/54b07684-e07c-452d-8b1d-870ea7f0a7b2" />        <img width="304" alt="image" src="https://github.com/user-attachments/assets/4431974c-bc9b-4819-9851-ac16d18f2e9a" />
<br><br><br>


**ML Models Userd**<br><br>
Baseline Model - Linear Regression with features used determined by Recursive Feature Elimination (RFE).<br>
Advanced Model - Neural Network with L2 Regularization and Grid Search CV for hyperparameter tuning.<br><br><br>


**📈 Results**<br>
The model's performance was evaluated using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). Detailed results and visualizations are available in the notebooks directory.<br><br>


**🚀 Deployment**<br>
The project is deployed using Streamlit and can be accessed here: https://groverrohit-bs-capstone-predicting-mag7-stockprices.streamlit.app/<br><br>


**🗂️ Repository Structure**<br>
data/: Contains datasets used in the project.<br>
notebooks/: Jupyter notebooks with analysis and model development.<br>
docs/: Documentation and reports.<br>
requirements.txt: Python dependencies.<br><br>


**📜 License**<br>
This project is licensed under the GPL-3.0 License.<br><br>


**📬 Contact**<br>
For questions or collaboration:<br><br>


Email: **groverrohit.m@gmail.com**<br>
LinkedIn: **https://www.linkedin.com/in/groverohit/**
