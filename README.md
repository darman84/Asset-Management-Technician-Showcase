Have a look at notebook.ipynb to see the finished model.

This research paper presents a proof of concept on the application of machine learning techniques to predict the timing of maintenance and inspection requirements for fire hydrant assets within the City of Plano. Utilizing historical data from the Cartegraph Asset Management System, this study aims to demonstrate how predictive analytics can help prioritize maintenance efforts by identifying hydrants that are likely to require attention sooner. This approach is crucial for enhancing public safety and optimizing maintenance resources. The motivation for this study stemmed from a job posting for an Asset Management Technician with the City of Plano, where I believed my skills could significantly contribute to the team. For a detailed analysis, please refer to the full paper. Below is a brief overview of the process:

1. **Data Extraction:**
    - Historical data for fire hydrants was downloaded from the City of Mesa's website into a JSON file.

2. **Feature Selection:**
    - From an initial set of 63 features, I selected the 7 most relevant to the asset's condition:
        - **'psi'**: Pressure level in pounds per square inch.
        - **'depth'**: Depth to the nut on the fire hydrant valve.
        - **'city_section'**: City section where the hydrant is located.
        - **'fire_hydrant_make'**: Brand of the hydrant.
        - **'fire_hydrant_model'**: Model of the hydrant.
        - **'hydrant_size'**: Size of the hydrant's barrel.
        - **'asb_year'**: Year the feature record drawing was completed.

3. **Algorithm Selection:**
    - The choice of algorithm was influenced by several factors:
        - Large and skewed dataset.
        - Mix of categorical and continuous values.
        - Presence of missing values and imbalanced classes.
        - Need for computational efficiency.
        - Nature of the task as a regression problem.
    - The **histogram gradient boosting regressor algorithm** was selected, with **random forest** as a secondary choice.

4. **Data Preprocessing:**
    - Missing values in categorical features were filled with the most common class, and those in continuous features with the median.
    - Continuous features were standardized to have a mean of 0 and a standard deviation of 1.

5. **Model Training and Testing:**
    - The data was split into 80% for training and 20% for testing.
    - GridSearchCV was utilized to optimize the model parameters.

6. **Model Evaluation:**
    - The model's performance was evaluated and further analyzed using SHAP values to understand feature influence.

7. **Key Findings:**
    - **'psi'** and **'years_since_added'** were highly influential, with higher psi values and fewer years since addition correlating with longer periods until maintenance was required.
    - **'depth'** had a lesser impact, with lower values correlating with less frequent maintenance.
    - Maintenance requirements varied significantly across different city sections, with sections **45** and **47** requiring less maintenance, while sections **22** and **103** required more.

This simplified breakdown provides an overview of the methodology and key findings of the research. For more detailed information, please refer to the full paper.

