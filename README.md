# Stroke-prediction-Capstone
A predictive machine learning model for early stroke detection and risk assessment using patient health data.
## Project Overview

Stroke is one of the leading causes of death and disability worldwide. 
Early detection and understanding of risk factors can help in prevention and treatment.

This project analyzes a healthcare stroke dataset to identify patterns and relationships 
between different health indicators and the occurrence of stroke.

Using Python data analysis tools, exploratory data analysis (EDA) was conducted to 
visualize patterns, understand risk factors, and generate insights from the dataset.

## Dataset Details
The analysis uses the `healthcare-dataset-stroke-data.csv` dataset, which contains 5,110 patient records. 

**Key Features:**
* **Demographics:** `gender`, `age`, `ever_married`, `work_type`, `Residence_type`
* **Health Metrics:** `bmi` (Body Mass Index), `avg_glucose_level`
* **Medical History:** `hypertension`, `heart_disease`, `smoking_status`
* **Target Variable:** `stroke` (1 = Patient suffered a stroke, 0 = No stroke)

*Note: Missing values in the `bmi` column were handled using median imputation to maintain data integrity.*
## Technologies & Libraries Used
* **Language:** Python 3
* **Environment:** Jupyter Notebook
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Logistic Regression, Random Forest, Gradient Boosting, SVC, Decision Trees, KNN), XGBoost
* **Data Preprocessing:** StandardScaler, LabelEncoder, SMOTE (Synthetic Minority Over-sampling Technique) for handling class imbalance.
## Methodology
The project followed a structured machine learning pipeline:
1. **Data Preprocessing:** Handled missing values (median imputation for `bmi`), encoded categorical variables using `LabelEncoder`, and scaled numerical features with `StandardScaler`.
2. **Handling Class Imbalance:** Applied SMOTE (Synthetic Minority Over-sampling Technique) to ensure the models did not become biased toward the majority class (patients without a stroke).
3. **Model Training & Evaluation:** Trained multiple classification algorithms and evaluated them based on predictive accuracy. 
4. **Feature Importance Analysis:** Extracted the key drivers of stroke prediction using the Gradient Boosting framework to understand the underlying clinical indicators.

## Exploratory Data Analysis (EDA)
Before training predictive models, an exploratory analysis was conducted to uncover the underlying distributions and relationships between patient attributes and stroke occurrence. 

### 1. Age Distribution Analysis
* **Observation:** The density plot analysis illustrates that stroke occurrences are predominantly concentrated in the older demographic, specifically between 60 and 85 years of age. Conversely, the non-stroke population is distributed across a much wider and younger range, with a higher density between 30 and 60 years.
* **Key Insight:** There is a strong positive relationship between advancing age and stroke risk. Age acts as a primary baseline indicator and is expected to be a highly influential predictor in the classification models.

### 2. Average Glucose Level Distribution
* **Observation:** There is a distinct divergence in average glucose levels between the two groups. Patients without a stroke history typically cluster around healthy baseline levels of 80–120 mg/dL. However, the stroke group tends to exhibit higher glucose levels, marked by a secondary density peak around 200 mg/dL.
* **Key Insight:** Elevated glucose levels are a significant risk marker. This distribution suggests that underlying metabolic conditions, such as diabetes or pre-diabetes, heavily compound the risk of a stroke.

### 3. The Nuance of Body Mass Index (BMI)
* **Observation:** The isolated BMI distributions for both stroke and non-stroke groups display significant overlap, with the vast majority of observations concentrated between 25 and 35. While stroke cases lean slightly toward higher BMI values, the visual distinction between the two classes is not sharply pronounced.
* **Key Insight:** When evaluated independently during EDA, BMI does not strongly differentiate the two classes, suggesting limited predictive power in isolation. *(Note: As detailed in the Feature Importance section, advanced tree-based models later identified BMI as a top predictor, indicating that its interaction with other variables like Age and Glucose is highly significant, even if it doesn't separate the classes linearly).*
### 4. Hypertension
* **Observation:** The majority of individuals in the dataset do not suffer from hypertension. However, when isolating stroke outcomes, those with hypertension exhibit a notably higher proportion of stroke cases compared to the non-hypertensive group.
* **Key Insight:** There is a clear positive association between hypertension and stroke occurrence, aligning with established medical consensus that high blood pressure is a major, direct risk factor.

### 5. Heart Disease
* **Observation:** Similar to hypertension, most individuals in the dataset do not have a history of heart disease. Nevertheless, the proportion of stroke cases is significantly higher among those who do. 
* **Key Insight:** Pre-existing heart disease substantially increases the likelihood of a stroke, making it a critical baseline medical feature for predictive modeling.

### 6. Gender
* **Observation:** The dataset contains a higher number of female participants than male participants. However, stroke cases are distributed across both genders in relatively similar proportions.
* **Key Insight:** Gender does not exhibit a strong direct association with stroke occurrence in this dataset, indicating it will likely have a weaker predictive contribution compared to cardiovascular indicators.

### 7. Smoking Status
* **Observation:** The majority of the dataset falls into the "never smoked" or "unknown" categories. Interestingly, a relatively higher proportion of stroke cases appears among individuals who "formerly smoked" compared to current smokers—likely because individuals often quit smoking after developing preliminary health complications. The "unknown" category introduces some noise into the dataset.
* **Key Insight:** Smoking behavior contributes to stroke risk, but its effect in this dataset is moderate when compared to dominant medical factors like age, glucose levels, and heart disease. 

### 8. Work Type
* **Observation:** The "private sector" category dominates the dataset, followed by "self-employed" and "government" roles. Categories like "children" and "never worked" contain almost zero stroke cases. 
* **Key Insight:** Work type itself is not a direct biological cause of stroke. Instead, it acts as a proxy variable for age and socioeconomic status. For example, the "children" category naturally represents a younger demographic with a near-zero stroke probability, while "private" and "self-employed" represent the aging adult workforce.

### 9. Residence Type
* **Observation:** The dataset shows a nearly 50/50 split between urban and rural residents. Stroke cases are also distributed very evenly between these two environments, with urban areas showing only a negligible increase.
* **Key Insight:** Residence type (urban vs. rural) demonstrates limited predictive power for stroke occurrence and is overshadowed by specific health and lifestyle metrics.
  ### Final Conclusion
* **Continuous Distribution:** Initial observation of the continuous BMI distributions showed significant overlap between the two groups (mostly concentrated between 25 and 35), making it difficult to sharply differentiate stroke and non-stroke cases at a glance.
* **Categorical Breakdown:** However, when segmented into standard BMI categories, a stark contrast emerges:
  * **Stroke Cases:** Overwhelmingly concentrated in higher BMI brackets. **Overweight (46.2%)** and **Obese (39.4%)** make up a combined **85.6%** of all stroke cases. Normal weight (14.1%) and Underweight (0.4%) are exceptionally rare.
  * **Non-Stroke Cases:** Exhibit a broader, more balanced spread. While Obese (37.5%) and Overweight (30.8%) are still prominent, there is a much larger share of Normal weight (24.8%) and Underweight (6.9%) individuals.
* **Key Insight:** Categorical analysis reveals that elevated BMI (specifically in the overweight and obese ranges) is strongly linked to increased stroke occurrence. This granular finding perfectly contextualizes why advanced machine learning models later identified BMI as the single most critical predictor of stroke risk in this dataset.
## Model Performance
Seven different classification models were trained and tested. Ensemble methods significantly outperformed traditional linear models for this dataset.

| Model | Accuracy Score |
| :--- | :--- |
| **XGBClassifier** | **0.93** |
| **RandomForestClassifier** | **0.92** |
| GradientBoostingClassifier | 0.91 |
| DecisionTreeClassifier | 0.88 |
| KNeighborsClassifier | 0.82 |
| SVC | 0.79 |
| LogisticRegression | 0.74 |

**Top Performer:** The `XGBClassifier` achieved the highest accuracy at **93%**, closely followed by the `RandomForestClassifier` at **92%**. These models demonstrate strong predictive capability and are highly suited for this diagnostic classification task.

## Key Findings & Feature Importance
To make the model interpretable for healthcare professionals, feature importance was extracted. The analysis revealed that demographic and metabolic factors are the strongest predictors of stroke risk.

**Top 3 Stroke Indicators:**
1. **BMI (0.291):** Body Mass Index emerged as the most significant predictor in the dataset.
2. **Age (0.219):** Advancing age is the second most critical risk factor.
3. **Average Glucose Level (0.211):** Blood sugar levels strongly correlate with stroke likelihood.

*Secondary Factors:* Smoking status (0.078), Residence type (0.054), and Work type (0.043) also contributed to the model, while baseline conditions like existing heart disease (0.026) and hypertension (0.031) had a surprisingly lower relative weight in this specific dataset's tree-based splits.

## Clinical Value
By identifying **BMI, Age, and Glucose Levels** as the primary drivers, this model can be integrated into early-warning health systems. Clinicians and preventative care platforms can use these insights to flag high-risk patients for targeted interventions, lifestyle coaching, or further medical screening before a critical event occurs.

---

## Installation & Usage
To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
