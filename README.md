# Code for the paper: `Predicting the longevity of resources shared in scientific publications`

***Experiment steps***
1. Preliminary exploration - Logistic regression, this is the benchmark. 
2. Feature selection - Lasso regression. 
3. Examine more traditional models, Elastic Net regression. 
4. Use the build-in Tobit model in R to experiment with the dataset. 
5. Rebuild the data(using the scripts in the ``data`` folder). 
6. Run the Random forest classifier to get the feature importance. 
7. Run Lasso regression to get the feature importance. 
8. Run Elastic Net regression to see its R-squared. 
9. Introduce the Elastic Net to the Tobit model. 
10. Hyper-tune the Tobit model. 
11. Explore the Tobit model.

## Folder ``archived``
Scripts used in the preliminary exploration stage. 

### ``archived/data_migration``
- Scripts for migrating data from MongoDB into Spark. 

### ``archived/glm``
- Modeling scripts which include Lasso/Logistic/Elastic Net Regression.  

### ``archived/tobit``
- Modeling scripts using Tobit censored regression. The script in the folder ``archived/tobit/spark`` is the Spark implementation of the Tobit regression. This version is not working. Do not spend time on it.  

### Folder ``archived/R``
- R implementation of Tobit censored regression.   

## Folder ``iconference_followup_study``
Scripts that are used in the paper. 

### Folder ``iconference_followup_study/data``
- ```iconference_followup_study/data/datacheck_data.ipynb``` To check the alive/dead ratio of URLs. 
- ```iconference_followup_study/data/data_proportion_inspection.ipynb``` To check the alive/dead proportion of URLs in the spreadsheet manner. 
- ```iconference_followup_study/data/data_truncated.ipynb``` To generate the dataset that contains truncated records(Selecting for dead URLs only). 
- ```iconference_followup_study/data/data_untruncated.ipynb``` To generate the dataset that contains truncated and untruncated records(Selecting for alive/dead URLs). The categorical ordinal feature `charset` is encoded by the frequency indexer. The script will standardize the whole dataset.  
- ```iconference_followup_study/data/data_untruncated_charset_viz.ipynb``` In addition to the previous dataset, this script selects the raw URL along with other variables. It is used to analyze the polarity of the URL longevity.

***There are few other data files on the server ```ist-deacuna-n1.syr.edu```***
1. ```/home/jjian03/iconference_followup_study/data/trunc_data.csv```
2. ```/home/jjian03/iconference_followup_study/data/trunc_data_cleaned.csv```
3. ```/home/jjian03/iconference_followup_study/data/untrunc_data.csv```
4. ```/home/jjian03/iconference_followup_study/data/untrunc_data_backup.csv```
5. ```/home/jjian03/iconference_followup_study/data/untrunc_data_cleaned.csv```
6. ```/home/jjian03/iconference_followup_study/data/untrunc_data_cleaned_url.csv```


### Folder ``iconference_followup_study/lasso``
- ```iconference_followup_study/lasso/lasso_truncated.ipynb``` To build the lasso model with truncated data(Dead URLs) only.  
- ```iconference_followup_study/lasso/lasso_untruncated.ipynb``` To build the lasso model with truncated/untruncated data(All the URLs). The categorical ordinal feature `charset` is encoded by the frequency indexer. The script will standardize the whole dataset.  
- ```iconference_followup_study/lasso/lasso_untruncated_cleaned.ipynb``` To build the lasso model with truncated/untruncated data(All the URLs). The categorical ordinal feature `charset` is encoded by the frequency indexer. The script will standardize the whole dataset.  
- ```iconference_followup_study/elastic_net/elastic_net_untruncated_cleaned.ipynb``` To build the elastic net model with truncated/truncated data(All the URLs). The categorical ordinal feature `charset` is encoded by the frequency indexer. The script will standardize the whole dataset. 

### Folder ``iconference_followup_study/elastic_net``
- ```iconference_followup_study/elastic_net/lib``` The dependent library of the elastic net analysis.  
- ```iconference_followup_study/elastic_net/elastic_net_truncated.ipynb``` To build the elastic net model with truncated data(Dead URLs) only. 
- ```iconference_followup_study/elastic_net/elastic_net_untruncated.ipynb``` To build the elastic net model with truncated/truncated data(All the URLs). The categorical ordinal feature `charset` is encoded by the frequency indexer. As it uses the package Plotly to visualize the data, the result of this script is very huge.  

### Folder ``iconference_followup_study/random_forest``
- ```iconference_followup_study/random_forest/best_lambda_scrutinize.ipynb``` To hyper-tune the parameters of the model random forest. 
- ```iconference_followup_study/random_forest/best_lambda_summary.ipynb``` To report the performance of the model random forest on its best parameters. 

### Folder ``iconference_followup_study/tobit``
- ```iconference_followup_study/tobit/eda``` Scripts for exploratory data analysis. 
- ```iconference_followup_study/tobit/grid_search``` Scripts searching the best hyper-parameters.
