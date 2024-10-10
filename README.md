# Analytic code for the paper: `Predicting the longevity of resources shared in scientific publications`
## Experiment Workflow
1. **Preliminary exploration**<br>
Begin with a baseline model using Logistic Regression to understand the initial performance on the dataset. 
2. **Feature selection**<br>
Apply Lasso regression for feature selection, helping to identify the most impactful features by enforcing sparsity
3. **Classic ML Model Exploration**<br>
Test Elastic Net regression, which combines both L1 (Lasso) and L2 (Ridge) regularization, to assess its performance in comparison to the baseline.
4. **Experiment with Tobit Model**<br>
Utilize the built-in Tobit model in R to handle censored data and explore how it fits the dataset.
5. **Data Preprocessing**<br>
Rebuild the dataset using the scripts provided in the `data` folder, ensuring that it is formatted correctly for subsequent experiments.
6. **Feature Importance Analysis - Random Forest**<br>
Run a Random Forest classifier to evaluate feature importance and compare the results with those obtained from regression models.
7. **Feature Importance Analysis - Lasso Regression**<br>
Re-run Lasso regression to calibrate feature importance based on the penalized regression method.
8. **Performance Evaluation - Elastic Net**<br>
Evaluate the performance of the Elastic Net regression by calculating its R-squared value to measure how well the model explains the variance in the dataset.
9. **Performance Evaluation - Tobit Model with Elastic Net Enhancement**<br>
Introduce the Elastic Net regularization into the Tobit model, combining censoring handling with regularization.
10. **Hyperparameter Tuning of the Tobit Model**<br>
Perform hyperparameter tuning on the Tobit model to optimize its performance.
11. **In-depth Exploration of Tobit Model**<br>
Conduct a detailed exploration of the Tobit model to understand its behavior and limitations on the dataset.
## Code Listing
This section serves as a guide to help you to navigate and understand the project's components. 
It provides an overview of the project's code organization. 
In this section, we include detailed explanations of the scripts' purposes in project sub-folders, covering data migration, modeling techniques (such as Lasso, Logistic, and Tobit regression), and analyses for this study.
### Folder `archived`
Scripts utilized during the preliminary exploration phase.
#### `archived/data_migration`
- Scripts for migrating data from MongoDB to Spark.
#### `archived/glm`
- Modeling scripts that include Lasso, Logistic, and Elastic Net regression.
#### `archived/tobit`
- Modeling scripts that utilize Tobit censored regression. The script located in the `archived/tobit/spark` folder is the Spark implementation of Tobit regression, but this version is currently non-functional. Please do not spend time on it.
#### `archived/R`
- R implementation of Tobit censored regression.
### Folder `iconference_followup_study`
Scripts utilized in the paper.
### Folder `iconference_followup_study/data`
- `iconference_followup_study/data/datacheck_data.ipynb`<br>
To assess the ratio of alive to dead URLs.
- `iconference_followup_study/data/data_proportion_inspection.ipynb`<br>
To analyze the proportion of alive to dead URLs in a spreadsheet format. 
- `iconference_followup_study/data/data_truncated.ipynb`<br>
To generate a dataset containing only truncated records (selecting for dead URLs).
- `iconference_followup_study/data/data_untruncated.ipynb`<br>
To generate a dataset that includes both truncated and untruncated records (selecting for alive and dead URLs). The categorical ordinal feature `charset` is encoded using a frequency indexer, and the script will standardize the entire dataset.  
- `iconference_followup_study/data/data_untruncated_charset_viz.ipynb`<br>
In addition to the previous dataset, this script selects the raw URLs along with other variables to analyze the polarity of URL longevity.<br>
### Folder `iconference_followup_study/lasso`
- `iconference_followup_study/lasso/lasso_truncated.ipynb`<br>
To build a Lasso model using only truncated data (dead URLs).  
- `iconference_followup_study/lasso/lasso_untruncated.ipynb`<br>
To build a Lasso model using both truncated and untruncated data (all URLs). The categorical ordinal feature `charset` is encoded with a frequency indexer, and the script will standardize the entire dataset.  
- `iconference_followup_study/lasso/lasso_untruncated_cleaned.ipynb`<br>
To build a Lasso model using both truncated and untruncated data (all URLs). The categorical ordinal feature `charset` is encoded using a frequency indexer, and the script will standardize the entire dataset.  
- `iconference_followup_study/elastic_net/elastic_net_untruncated_cleaned.ipynb`<br>
To build an Elastic Net model using both truncated and untruncated data (all URLs). The categorical ordinal feature `charset` is encoded with a frequency indexer, and the script will standardize the entire dataset.
### Folder `iconference_followup_study/elastic_net`
- `iconference_followup_study/elastic_net/lib`<br>
The dependent library for Elastic Net analysis.
- `iconference_followup_study/elastic_net/elastic_net_truncated.ipynb`<br>
To build an Elastic Net model using only truncated data (dead URLs). 
- `iconference_followup_study/elastic_net/elastic_net_untruncated.ipynb`<br>
To build an Elastic Net model using both truncated and untruncated data (all URLs). The categorical ordinal feature `charset` is encoded using a frequency indexer. Since the script utilizes the Plotly package for data visualization, the output can be quite large.
### Folder `iconference_followup_study/random_forest`
- `iconference_followup_study/random_forest/best_lambda_scrutinize.ipynb`<br>
To fine-tune the hyperparameters of the Random Forest model.
- `iconference_followup_study/random_forest/best_lambda_summary.ipynb`<br>
To report the performance of the Random Forest model with its best hyperparameters.
### Folder `iconference_followup_study/tobit`
- `iconference_followup_study/tobit/eda`<br>
Scripts for exploratory data analysis (EDA). 
- `iconference_followup_study/tobit/grid_search`<br>
Scripts for searching optimal hyperparameters.
