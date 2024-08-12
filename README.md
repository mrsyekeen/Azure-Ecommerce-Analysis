# Group Project: Azure Ecommerce business Analysis 


Welcome to our project! We have taken this project to express our knowledge of buisness intelligence on a kaggle UK online business dataset (https://www.kaggle.com/datasets/gabrielramos87/an-online-shop-business)

## Kaggle dataset context

A sales transaction data set of London based online retail e-commerce for one year. The company had customers all over the world that purchase products themselves. The dataset has 500K entries and 8 features, 42mb data size. The following is the description of each feature.

TransactionNo (categorical): a six-digit unique number that defines each transaction. The letter “C” in the code indicates a cancellation.

Date (numeric): the date when each transaction was generated.

ProductNo (categorical): a five or six-digit unique character used to identify a specific product.

Product (categorical): product/item name.

Price (numeric): the price of each product per unit in pound sterling (£).

Quantity (numeric): the quantity of each product per transaction. Negative values related to cancelled transactions.

CustomerNo (categorical): a five-digit unique number that defines each customer.

Country (categorical): name of the country where the customer resides.
There is a small percentage of order cancellation in the data set. Most of these cancellations were due to out-of-stock conditions on some products. Under this situation, customers tend to cancel an order as they want all products delivered all at once.

## Project Overview

In this project, our team focused on predicting inventory quantities for the next year and perform customer segmentation. We used various machine learning models, including Random Forest, XGBoost, and Linear Regression, to achieve the most accurate predictions possible and group the customers based on recency, frequency and monetary. The project was built using PySpark and Azure Databricks, which provided us with the scalability and power needed to handle our large dataset.

## Why This Project?

Inventory management is crucial for businesses. By accurately predicting future prodcut quantity, the company can optimize their stock levels, reduce costs, and ensure they meet customer demand without overstocking. Our project aims to address this challenge by building a robust prediction model. Also we aim to classify their customers based on activities on the website.

## Tools & Technologies
- **Azure Databricks**: To efficiently manage and run our PySpark code.
- **Machine Learning Models**: We used Random Forest, XGBoost, and Linear Regression to compare and find the best model for our predictions.
- **Pandas & NumPy**: For data manipulation and numerical computations.
- **Seaborn & Matplotlib**: To visualize our findings and model performance.
- **PySpark MLlib**: We used PySpark MLlib to build and train machine learning models like Random Forest and Linear Regression on distributed data.
- **Scikit-learn (sklearn)**: Employed for training the XGBoost model, performing hyperparameter tuning, and calculating evaluation metrics like RMSE and R2. sklearn offered a user-friendly interface for model training and evaluation.

## Our Approach

1. **Data Preprocessing**: We started by cleaning the data, handling missing values, and transforming the features into a suitable format for model training.
2. **Feature Engineering**: Using Principal Component Analysis (PCA), we reduced the dimensionality of our dataset to focus on the most relevant features for predicting inventory quantities.
3. **Model Training**: We trained multiple models and tuned their hyperparameters to find the best-performing one.
4. **Evaluation**: We evaluated our models using metrics like R2 and RMSE to ensure they met our accuracy standards.
5. **Prediction**: Finally, we used our trained models to predict future inventory quantities and compared their performance.
6. **Sillhoutte score**: To predict the possible number of clusters of customers.
7. **Clustering**: K-means was used for clustering.
8. **RFM**: RFM analysis was use to classify the clusters for marketing strategies.

## Git uploads
- EcomDAtaTransform-1.ipynb contains the code for data preprocessing, feature engineering, visualization, modeling and clustering. Here the transformed data (part-00000-tid-6505393303909980396-cd400216-3502-4fcc-98a3-2be5ab25b152-365-1-c000) was saved in Azure Datalake.
- EconBizVisual-1 contains the 3 pages of bussiness analytics
- Processed_data is a pickled preprocessed data
- Standardized pickle were uploaded in batches of 10k entries due to the size of dataset.
- Selected features were pickled as Transact_data.pkl
- XGBoost model was saved with joblib
## Results & Insights

Each model provided unique insights into the data, and we compared them to determine which one had better RMSE and R2 for our prediction goals. XGBoost performed metrics had the best with 97% accuracy while the customers are predicted to be in 3 classes named: Loyal, Ocassionally and at risk. With the prediction and customer classification, the company will optimize their inventory based on the spending power of thier customrers all over the world and imprement strategies to keep most loyal.

## Acknowledgments

We Kaggle for providing the data and overview, Meysam our instructor for his support, knowldge and direction on this project, above all, the entire team for making possible for us to learn how to use different platforms to achieve the project success.
