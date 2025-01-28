# bs-capstone-predicting-stockprices
Prediction of Stock Prices of Mag 7 stocks through Data Science

------------------------------------------------------------------------------

## Predicting Stock Prices of Mag-7 stocks
=========================

### Executive Summary

Predict the long-term performance of any large-cap stock over an year based on –
Fundamental Factors – EPS and P/E Ratio
Technical Factors – GDP Growth, Inflation, Unemployment Rate, Retail Sales and Industrial Production
News and Market Sentiment

##### US Stocks and the Mag-7 - 
A combined market capitalization of about $14 trillion
Exposure to high-growth technologies such as high-end software and hardware, cloud computing, and Artificial Intelligence
Each of the seven stocks has outperformed the S&P 500 by a huge margin in the past decade



##### The Data Science Approach - 
Gather last 10 years’ data from multiple sources and APIs
Clean and Pre-process data
Preliminary EDA
Machine Learning – Train a regression model on a subset of the data
Test the model on another subset of data
Predict the performance of the Mag-7 stocks over the next year

##### Impact -
Can help individual or institutional investors optimally manage their investments in stocks.

##### Description of Dataset -
Data is collected from various Apis and webpages downloaded from the internet.

#### Repository 

* `data` 
    - contains link to copy of the dataset (stored in a publicly accessible cloud storage)
    - saved copy of aggregated / processed data as long as those are not too large (> 10 MB)

* `model`
    - `joblib` dump of final model(s)

* `notebooks`
    - contains all final notebooks involved in the project

* `docs`
    - contains final report, presentations which summarize the project

* `references`
    - contains papers / tutorials used in the project

* `src`
    - Contains the project source code (refactored from the notebooks)

* `.gitignore`
    - Part of Git, includes files and folders to be ignored by Git version control

* `conda.yml`
    - Conda environment specification

* `README.md`
    - Project landing page (this page)
