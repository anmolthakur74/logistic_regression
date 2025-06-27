# Task 4: Logistic Regression on Heart Disease Dataset

This repository contains my solution for **Task 4** of the AI & ML Internship. The focus of this task is to build a logistic regression model for binary classification using the Heart Disease dataset and evaluate its performance using key metrics and visualizations.

---

## Objective

Build a binary classifier using logistic regression.

---

## Files Included

| File Name                     | Description                                                  |
|------------------------------|--------------------------------------------------------------|
| `heart.csv`                  | Dataset used for the classification task                    |
| `logistic_regression.ipynb`  | Jupyter Notebook with all steps: preprocessing to evaluation |
| `screenshots/`               | Folder containing ROC curves, sigmoid function plot, etc.    |
| `README.md`                  | Project documentation                                        |

---

## What I Did

1. I began with the basics of exploratory data analysis using simple pandas dataframe functions (e.g., df.head(), df.info(), df.describe()). This allowed me to get a feel for the data, check for missing values and assess how balanced the target variable was. From my initial analysis, I found that the target variable of people with heart disease and without did not show too much bias (526 with heart disease vs 499 without heart disease).

2. Next I extracted the input features (X) and target variable (y) with 80/20 training/test splits. At this point, I performed the split, so one set of data could be used for training, and the other could be used to test how well the model performed on unseen data.

3. Prior to training, I transformed the numerical features using StandardScaler to standardize the feature. It is important to standardize features that span different orders of magnitude, such as age, cholesterol level and resting blood pressure. Standardization will allow the features to be input to the model in the same numerical format and will ensure the model does not place too much value on features with larger numbers.

4. From sklearn, I used scikit-learn’s LogisticRegression method to train a logistic regression model. Training the model by using logistic regression serves as a good baseline for a general binary classification tasks by predicting the presence or absence of heart disease.

5. In order to assess the model I looked at multiple metrics: 

- A confusion matrix to see how many predictions were correct and incorrect. 

- Precision to assess how many of the positive predictions were correct. 

- Recall to measure how many of the actual positives were captured correctly. 

- ROC-AUC score, to visualize the model's ability to separate the two classes.

- And finally, I plotted the ROC curve using matplotlib to get a sense of the model's performance.

6. I also played around with changing the decision threshold from .05 to .04. Lowering the decision threshold allowed the model to capture a greater number of positive cases (high recall) while potentially increasing the false positive rate (lower precision). In medical situations, this is valuable because it is better to declare someone a potential case than to miss a serious condition.

7. Lastly, I plotted the sigmoid function, which is the essence of how logistic regression functions. The sigmoid function takes in any number (positive or negative) and squashes that into a number between 0 and 1. In this way, the model spits out probabilities, rather than binary predictions.

---

## Tools & Libraries Used

- Python 3.12
- pandas, numpy — for data handling
- matplotlib — for data visualization
- scikit-learn — for preprocessing, modeling, and evaluation

---

## What I Learned

This task taught me how to build a classification model from scratch using logistic regression. I learned the importance of feature scaling, threshold tuning, and model evaluation using metrics beyond accuracy. I also understood how logistic regression leverages the sigmoid function to output probabilities.

---

## How to Run This Project

1. Clone the repository:
   ```bash
   git clone https://github.com/anmolthakur74/task-4-logistic-regression.git
   cd task-4-logistic-regression
   ```
   
2. Install required libraries:
    ```bash
    pip install pandas numpy matplotlib scikit-learn
    ```
    
3. Open the notebook:
   ```bash
    jupyter notebook logistic_regression.ipynb
    ```

---

## Author

**Anmol Thakur**

GitHub: [anmolthakur74](https://github.com/anmolthakur74)
