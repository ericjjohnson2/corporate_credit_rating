# corporate_credit_risk

**Group Details:**
- Group 3 Members: Mitchell Lor, Frewoini Mebrahtu, Eric Johnson, Lucinda Hodgson

**Project Title:** Predicting Corporate Credit Ratings

**Data Source:** Kaggle
[Dataset Link](https://www.kaggle.com/datasets/kirtandelwadia/corporate-credit-rating-with-financial-ratios)

**Overview:**

**Introduction:**
The project aims to leverage data analysis techniques to extract meaningful insights and predict credit ratings for corporations to assist in investment decisions. By utilizing a dataset sourced from Kaggle, the group intends to preprocess the data meticulously before applying various machine learning models for predictive analysis. The models will undergo optimization and evaluation to ensure accuracy and reliability in predicting credit ratings.

**Project Details:**

1. **Data Acquisition and Preprocessing:**
   - A robust dataset comprising over 7000 records was sourced from Kaggle.
   - The data was loaded into a database using Python Pandas, followed by SQL queries for data retrieval.
   - Cleaning and preprocessing involved dropping unnecessary columns and identifying significant metrics for analysis.
  
     ![binaryimage](img/agencies/RatingAgency.png)

2. **Initial Attempts:**
   - Initially, deep learning techniques were explored; however, encountered roadblocks due to overfitting and imbalanced data.
   - Overfitting was observed due to the simplicity of the data, leading to poor generalization.
   - Imbalanced data, where investment grade loans dominated, posed challenges for deep learning.

3. **Model Evaluation:**
   - Three models were developed and evaluated:
     - Model 1: Loss - 0.636, Accuracy - 0.667
     - Model 2: Loss - 0.490, Accuracy - 0.791
     - Model 3: Loss - 0.439, Accuracy - 0.797

   ![deeplearning](img/models/deeplearning/model5_plot.png)

5. **Random Forest Model for Credit Rating Forecasting:**
   - A Random Forest Classifier model was employed to forecast credit ratings based on a curated dataset.
   - Data preprocessing involved loading and cleaning data, extracting essential features, and incorporating dummy variables for categorical data representation.
   - The dataset was split into training and testing sets, and standard scaling was applied for consistent feature scaling.
   - A Random Forest Classifier with 500 decision trees was trained on the scaled data to capture complex relationships.
  
     
      ![Random Forest Confusion Matrix](img/models/random_forest/model5_confusion_matrix.png)

      ![Random Forest Importances Plot](img/models/random_forest/model5_importances_plot.png)

      ![Random Forest ROC Curve](img/models/random_forest/model5_roc_curve.png)

      [PDF Example of a Random Tree in Our Model](img/models/random_forest/model5_random_tree.pdf)

6. **Model Evaluation and Feature Importance Analysis:**
   - The model's performance was evaluated using standard metrics such as confusion matrix, accuracy score, and classification report.
   - Additionally, a feature importance analysis was conducted to identify the significant contributors to credit rating prediction.

7. **Search API to Test Model:**
   - Using Alpha Vantage Data
   - Pulling API based on Ticker values and feeding their recent financial performance into the models.
   
     ![preview](img/preview.gif)


## **Conclusion:**
The application of machine learning has yielded encouraging outcomes. Through experimentation with various models, some patterns have emerged: certain models excel in predicting positive outcomes, while others are proficient in identifying negative outcomes. The random forest models are the top performers with 95% accuracy rate on this test dataset.

However, during deployment in real-world scenarios, particularly in predicting junk credit status (S&P BB+ or lower), challenges arose. Despite techniques like oversampling and undersampling to address class imbalances, the models struggled to accurately identify instances of junk credit. They did however exhibit consistent success in predicting good credit status.

To enhance model performance, alternative methods were explored such as k-folding and feature engineering. One notable limitation was the absence of industry sector information in our API. This was available in training and testing datasets, and when utilized the model performance improved. But these features were dropped due to constraints in the API's data retrieval capabilities. It is evident that incorporating industry sector data could significantly enhance prediction accuracy.


