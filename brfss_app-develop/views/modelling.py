

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor,GradientBoostingClassifier,AdaBoostRegressor,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge, BayesianRidge, Lasso
from sklearn.metrics import mean_squared_error, accuracy_score,mean_absolute_error
import streamlit as st
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

def load_view(data):
    st.title("Modelling")

    # Define algorithm explanations with strengths and weaknesses
    algorithm_descriptions = {
        'KNN': {
            'type': 'Classifier/Regressor',
            'description': """**K-Nearest Neighbors (KNN)** is a non-parametric, instance-based learning algorithm. It classifies data points based on their proximity to their nearest neighbors.""",
            'strengths': """   
                - Easy to understand and implement  
                - Works well with small datasets and non-linear data""",
            'weaknesses': """  
                - Computationally expensive with large datasets  
                - Sensitive to the choice of 'K' and to noise"""
        },
        'SVM': {
            'type': 'Classifier/Regressor',
            'description': """Support Vector Machine (SVM) is a powerful algorithm that works by finding the hyperplane that best separates the data.""",
            'strengths': """   
                - Effective in high-dimensional spaces, works well with clear margin separation.""",
            'weaknesses': """  
                - Computationally expensive, not well-suited for large datasets or overlapping classes, sensitive to kernel selection."""
        },
        'Decision Tree': {
            'type': 'Classifier/Regressor',
            'description': """Decision Trees are intuitive models that split data into branches to make predictions.""",
            'strengths': """   
                - Easy to interpret and visualize, works with both classification and regression tasks.""",
            'weaknesses': """  
                - Prone to overfitting, especially with small datasets, sensitive to small data variations."""
        },
        'Naive Bayes': {
            'type': 'Classifier',
            'description': """Naive Bayes is based on applying Bayes’ theorem with a strong assumption of independence between features.""",
            'strengths': """   
                - Fast, efficient for large datasets and text classification.""",
            'weaknesses': """  
                - Assumes feature independence, which is often unrealistic; performs poorly when features are highly correlated."""
        },
        'Random Forest': {
            'type': 'Classifier/Regressor',
            'description': """Random Forest is an ensemble learning method that builds multiple decision trees and combines their outputs.""",
            'strengths': """   
                - Reduces overfitting, works well with high-dimensional data.""",
            'weaknesses': """  
                - Can be slower to train and interpret, may not perform well on highly sparse data."""
        },
        'Linear Regression': {
            'type': 'Regressor',
            'description': """Linear Regression models the linear relationship between independent and dependent variables.""",
            'strengths': """   
                - Simple to implement, interpretable, and works well when data has a linear trend.""",
            'weaknesses': """  
                - Struggles with non-linear data and outliers, sensitive to multicollinearity."""
        },
        'Logistic Regression': {
            'type': 'Classifier',
            'description': """Logistic Regression models the probability of a binary outcome based on one or more independent variables.""",
            'strengths': """   
                - Simple and interpretable, works well for binary classification tasks with linearly separable data.""",
            'weaknesses': """  
                - Performs poorly with non-linear relationships, sensitive to outliers and multicollinearity."""
        },
        'Ridge Regression': {
            'type': 'Regressor',
            'description': """Ridge Regression adds L2 regularization to linear regression to reduce overfitting.""",
            'strengths': """   
                - Reduces overfitting and handles multicollinearity well.""",
            'weaknesses': """  
                - Lacks feature selection capabilities, struggles with non-linear data."""
        },
        'Gradient Boosting': {
            'type': 'Classifier/Regressor',
            'description': """Gradient Boosting builds models sequentially, optimizing the errors made by prior models.""",
            'strengths': """   
                - Highly accurate, handles non-linear relationships well, great for imbalanced datasets.""",
            'weaknesses': """  
                - Slow to train, prone to overfitting if not tuned properly, requires careful tuning of parameters."""
        },
        'Bayesian Ridge Regression': {
            'type': 'Regressor',
            'description': """Bayesian Ridge Regression incorporates Bayesian inference into linear regression.""",
            'strengths': """   
                - Provides uncertainty estimates, good for small datasets.""",
            'weaknesses': """  
                - Computationally expensive, assumes linearity and Gaussian noise."""
        },
        'Lasso Regression': {
            'type': 'Regressor',
            'description': """Lasso Regression adds L1 regularization to linear regression, performing feature selection by shrinking coefficients to zero.""",
            'strengths': """   
                - Useful for feature selection and reducing model complexity.""",
            'weaknesses': """  
                - Can ignore important features if they are correlated, struggles with multicollinearity."""
        },
        'Neural Network': {
            'type': 'Classifier/Regressor',
            'description': """Neural Networks are a set of algorithms modeled after the human brain, capable of capturing complex relationships.""",
            'strengths': """   
                - Excellent for handling non-linear data and complex relationships, works well with large datasets.""",
            'weaknesses': """  
                - Requires a large amount of data and computation, prone to overfitting without proper regularization, hard to interpret."""
        }
    }

    # Step1: Selecting algorithm
    algorithm = st.selectbox("Select Supervised Machine Learning Algorithm",
                            ("KNN", "SVM", "Decision Tree", "Naive Bayes", "Random Forest", "Linear Regression", 
                            "Logistic Regression", "Ridge Regression", "Gradient Boosting", "Bayesian Ridge Regression", 
                            "Lasso Regression", "Neural Network"))

    # Step2: Selecting regressor or classifier
    if algorithm in ['Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Bayesian Ridge Regression']:
        algorithm_type = "Regressor"
        st.sidebar.write(f"{algorithm} only does Regression.")
    elif algorithm in ['Naive Bayes', 'Logistic Regression']:
        algorithm_type = "Classifier"
        st.sidebar.write(f"{algorithm} only does Classification.")
    else:
        # For models that can be both, let the user decide
        algorithm_type = st.selectbox("Select Algorithm Type", ("Classifier", "Regressor"))

    # Display algorithm description
    if algorithm in algorithm_descriptions:
        # st.sidebar.write(f"**{algorithm}** is used for {algorithm_type}.")
        st.sidebar.write(f"{algorithm_descriptions[algorithm]['description']}")
        st.sidebar.write(f"**Strengths:**{algorithm_descriptions[algorithm]['strengths']}")
        st.sidebar.write(f"**Weaknesses:**{algorithm_descriptions[algorithm]['weaknesses']}")

        # Function to extract data from HTML
    def extract_data_from_html(file_path):
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        cells = soup.find_all('td', {'class': 'l m linecontent'})
        data = []

        for cell in cells:
            parts = cell.decode_contents().split('<br/>')
            row_data = {}
            for part in parts:
                part = part.replace('\xa0', ' ').strip()
                if 'Label:' in part:
                    row_data['Label'] = part.split('Label:')[1].strip()
                if 'Question:' in part:
                    row_data['Question'] = part.split('Question:')[1].strip()
                if 'SAS Variable Name:' in part:
                    row_data['SAS Variable Name'] = part.split('SAS Variable Name:')[1].strip()

            if row_data:
                data.append(row_data)

        return pd.DataFrame(data)

    # Extracting the HTML data
    html_data = extract_data_from_html('information.HTML')

    def input_output(data):
        # Normalize DataFrame column names
        data.columns = data.columns.str.strip()

        # Filter HTML data to only include columns present in DataFrame
        available_columns = data.columns
        filtered_html_data = html_data[html_data['SAS Variable Name'].isin(available_columns)]

        # Create mapping of options with labels and SAS Variable Names
        options_with_labels = {
            f"{row['SAS Variable Name']} ({row['Label']})": row['SAS Variable Name'].strip()
            for _, row in filtered_html_data.iterrows()
        }

        # Select X columns
        selected_x_columns_with_labels = st.multiselect("Select At Least One Feature (X)", options_with_labels.keys())
        
        # Map back to the actual SAS variable names
        selected_x_columns = [options_with_labels[label] for label in selected_x_columns_with_labels]

        # Select Y column, using the same filtering and mapping logic
        y_options_with_labels = {key: value for key, value in options_with_labels.items() if value in available_columns}
        selected_y_column_with_label = st.selectbox("Select Target Variable (Y)", y_options_with_labels.keys())
        selected_y_column = y_options_with_labels[selected_y_column_with_label]


        # Show corresponding information for selected X columns
        if selected_x_columns:
            st.write("**Selected Features Information:**")
            for col in selected_x_columns:
                info = html_data[html_data['SAS Variable Name'] == col]
                if not info.empty:
                    st.write(f"**{col}**: Label: {info['Label'].values[0]}, Question: {info['Question'].values[0]}")

        # Show corresponding information for selected Y column
        predict_name = None
        if selected_y_column:
            st.write("**Selected Target Information:**")
            info = html_data[html_data['SAS Variable Name'] == selected_y_column]
            if not info.empty:
                predict_name = info['Label'].values[0]
                st.write(f"**{selected_y_column}**: Label: {info['Label'].values[0]}, Question: {info['Question'].values[0]}")

        # Print columns for debugging
        print(f"Selected X columns: {selected_x_columns}")
        print(f"Selected Y column: {selected_y_column}")
        print(f"Data columns: {data.columns}")

        X = data[selected_x_columns]
        Y = data[selected_y_column]

        return X, Y, selected_x_columns, predict_name
    
    # Step4-1: Adding Parameters For Classifier
    def add_parameter_classifier_general(algorithm):

        # Declaring a dictionary for storing parameters
        params = dict()

        # Add paramters for SVM ---Checked----
        if algorithm == 'SVM':

            # Add regularization parameter from range 0.01 to 10.0
            c_regular = st.sidebar.slider('C (Regularization)', 0.01, 10.0)
            # Add kernel is the arguments in the ML model
            # Polynomial ,Linear, Sigmoid and Radial Basis Function are types of kernals 
            kernel_custom = st.sidebar.selectbox('Kernel', ('linear', 'poly', 'rbf', 'sigmoid'))
            # Add parameters into dictionary
            params['C'] = c_regular
            params['kernel'] = kernel_custom

        # Adding Parameters for KNN ----Checked----
        elif algorithm == 'KNN':

            # Add Number of Neighbour (1-20) to KNN 
            k_n = st.sidebar.slider('Number of Neighbors (K)', 1, 20,key="k_n_slider")

            # Adding weights
            weights_custom = st.sidebar.selectbox('Weights', ('uniform', 'distance'))

            # Add parameters into dictionary
            params['K'] = k_n
            params['weights'] = weights_custom

        # Add Parameters for Naive Bayes ----Checked----
        # It doesn't have any paramter
        elif algorithm == 'Naive Bayes':
            st.sidebar.info("This is a simple algorithm. It doesn't have Parameters for hyperparameter tuning.")

        # Add Parameters for Decision Tree ----Checked----
        elif algorithm == 'Decision Tree':

            # Add max_depth
            max_depth = st.sidebar.slider('Max Depth', 2, 17)
            # Add criterion
            # mse is for regression (it is used in DecisionTreeRegressor)
            # mse will give error in classifier so it is removed
            criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy'))

            # Add splitter
            splitter = st.sidebar.selectbox("Splitter", ("best", "random"))

            # Add to dictionary
            params['max_depth'] = max_depth
            params['criterion'] = criterion
            params['splitter'] = splitter

            # Exception Handling using try except block
            # Because we are sending this input in algorithm model it will show error before any input is entered
            # For this we will do a default random state till the user enters any state and after that it will be updated
            try:
                random = st.sidebar.text_input("Enter Random State")
                params['random_state'] = int(random)
            except:
                params['random_state'] = 4567

        # Add Parameters for Random Forest ----Checked----
        elif algorithm == 'Random Forest':

            # Add max_depth
            max_depth = st.sidebar.slider('Max Depth', 2, 17)

            # Add number of estimators
            n_estimators = st.sidebar.slider('Number of Estimators', 1, 90)

            # Add criterion
            # mse is for regression (it is used in RandomForestRegressor)
            # mse will give error in classifier so it is removed
            criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy', 'log_loss'))


            # Add to dictionary
            params['max_depth'] = max_depth
            params['n_estimators'] = int(n_estimators)
            params['criterion'] = criterion

            # Exception Handling using try except block
            # Because we are sending this input in algorithm model it will show error before any input is entered
            # For this we will do a default random state till the user enters any state and after that it will be updated
            try:
                random = st.sidebar.text_input("Enter Random State")
                params['random_state'] = int(random)
            except:
                params['random_state'] = 4567

        # Adding Parameters for Logistic Regression ----Checked----
        elif algorithm == 'Logistic Regression':

            # Adding regularization parameter from range 0.01 to 10.0
            c_regular = st.sidebar.slider('C (Regularization)', 0.01, 10.0)
            params['C'] = c_regular
            # Taking fit_intercept
            fit_intercept = st.sidebar.selectbox("Fit Intercept", ('True', 'False'))
            params['fit_intercept'] = bool(fit_intercept)

            # Add Penalty 
            penalty = st.sidebar.selectbox("Penalty", ('l2', None))
            params['penalty'] = penalty

            # Add n_jobs
            n_jobs = st.sidebar.selectbox("Number of Jobs", (None, -1))
            params['n_jobs'] = n_jobs

            # Add solver
            solver = st.sidebar.selectbox("Solver", ('lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky','sag', 'saga'))
            params['solver'] = solver

        elif algorithm == 'Gradient Boosting':
            #Add loss
            loss = st.sidebar.selectbox("Loss",('log_loss','exponential'))
            params['loss'] = loss

            #Add n_estomators
            n_estimators = n_estimators = st.sidebar.slider('Number of Estimators', 1, 100)
            params['n_estimators'] = int(n_estimators)

            #Add learning rate
            learning_rate = st.sidebar.slider('Learning Rate', 0.01, 1.0)
            params['learning_rate'] = learning_rate


        elif algorithm == 'Neural Network':
            hidden_layer_sizes = st.sidebar.selectbox("Hidden Layer Sizes", ((100,), (50, 50), (100, 50)))
            params['hidden_layer_sizes'] = hidden_layer_sizes

            activation = st.sidebar.selectbox("Activation", ('logistic', 'tanh', 'relu'))
            params['activation'] = activation

            solver = st.sidebar.selectbox("Solver", ('lbfgs', 'sgd', 'adam'))
            params['solver'] = solver

            learning_rate_init = st.sidebar.slider('Learning Rate', 0.01, 1.0)
            params['learning_rate_init'] = learning_rate_init


        return params



    # Step4-2: Adding Parameters for regressor
    def add_parameter_regressor(algorithm):

        # Declaring a dictionary for storing parameters
        params = dict()

        # Deciding parameters based on algorithm
        # Add Parameters for Decision Tree  ----Checked----
        if algorithm == 'Decision Tree':

            # Add max_depth
            max_depth = st.sidebar.slider('Max Depth', 2, 17)

            # Add criterion
            # mse is for regression- It is used in DecisionTreeRegressor
            criterion = st.sidebar.selectbox('Criterion', ('absolute_error', 'squared_error', 'poisson', 'friedman_mse'))

            # Add splitter
            splitter = st.sidebar.selectbox("Splitter", ("best", "random"))

            # Adding to dictionary
            params['max_depth'] = max_depth
            params['criterion'] = criterion
            params['splitter'] = splitter

            # Exception Handling using try except block
            # Because we are sending this input in algorithm model it will show error before any input is entered
            # For this we will do a default random state till the user enters any state and after that it will be updated
            try:
                random = st.sidebar.text_input("Enter Random State")
                params['random_state'] = int(random)
            except:
                params['random_state'] = 4567

        # Adding Parameters for Linear Regression ----Checked----
        elif algorithm == 'Linear Regression':

            # Add fit_intercept
            fit_intercept = st.sidebar.selectbox("Fit Intercept", ('True', 'False'))
            params['fit_intercept'] = bool(fit_intercept)

            # Add n_jobs
            n_jobs = st.sidebar.selectbox("Number of Jobs", (None, -1))
            params['n_jobs'] = n_jobs

        # Add Parameters for Random Forest ----Checked----
        elif algorithm == 'Random Forest':

            # Add max_depth
            max_depth = st.sidebar.slider('Max Depth', 2, 17)

            # Add number of estimators
            n_estimators = st.sidebar.slider('Number of Estimators', 1, 90)

            # Add criterion
            # mse is for regression- It is used in RandomForestRegressor
            criterion = st.sidebar.selectbox('Criterion', ('absolute_error', 'squared_error', 'poisson', 'friedman_mse'))

            # Add to dictionary
            params['max_depth'] = max_depth
            params['n_estimators'] = n_estimators
            params['criterion'] = criterion

            # Exception Handling using try except block
            # Because we are sending this input in algorithm model it will show error before any input is entered
            # For this we will do a default random state till the user enters any state and after that it will be updated
            try:
                random = st.sidebar.text_input("Enter Random State")
                params['random_state'] = int(random)
            except:
                params['random_state'] = 4567

        # Add Parameter for Ridge ----Changed----
        elif algorithm == 'Ridge Regression':

            #Add ridge_aplha
            alpha = st.sidebar.slider("Alpha", 0, 2)
            params['alpha'] = alpha

        elif algorithm == 'Gradient Boosting':
            #Add loss
            loss = st.sidebar.selectbox("Loss",('squared_error','absolute_error','huber','quantile'))
            params['loss'] = loss


            #Add learning rate
            learning_rate = st.sidebar.slider('Learning Rate', 0.01, 1.0)
            params['learning_rate'] = learning_rate


            #Add n_estomators
            n_estimators = n_estimators = st.sidebar.slider('Number of Estimators', 1, 100)
            params['n_estimators'] = int(n_estimators)


        elif algorithm == 'Bayesian Ridge Regression':
            alpha_1 = st.sidebar.slider('Alpha 1', min_value=1e-6, max_value=1e-3, step=1e-5,format="%.1e")
            alpha_2 = st.sidebar.slider('Alpha 2', min_value=1e-6, max_value=1e-3, step=1e-5, format="%.1e")
            lambda_1 = st.sidebar.slider('Lambda 1', min_value=1e-6, max_value=1e-3, step=1e-5, format="%.1e")
            lambda_2 = st.sidebar.slider('Lambda 2', min_value=1e-6, max_value=1e-3, step=1e-5, format="%.1e")

            params['alpha_1'] = alpha_1
            params['alpha_2'] = alpha_2
            params['lambda_1'] = lambda_1
            params['lambda_2'] = lambda_2

        elif algorithm == 'Lasso Regression':
            alpha = st.sidebar.slider('Alpha', 0.01,2.0)
            params['alpha'] = alpha


        elif algorithm == 'Neural Network':
            hidden_layer_sizes = st.sidebar.selectbox("Hidden Layer Sizes", ((100,), (50, 50), (100, 50)))
            params['hidden_layer_sizes'] = hidden_layer_sizes

            activation = st.sidebar.selectbox("Activation", ('logistic', 'tanh', 'relu'))
            params['activation'] = activation

            solver = st.sidebar.selectbox("Solver", ('lbfgs', 'sgd', 'adam'))
            params['solver'] = solver

            learning_rate_init = st.sidebar.slider('Learning Rate', 0.01, 1.0)
            params['learning_rate_init'] = learning_rate_init



        return params

    #Step5
    # Calling Function based on regressor and classifier
    # Here since the parameters for regressor and classifier are same for some algorithm we can directly use this
    # Because of this here except for this three algorithm we do not need to take parameters separately


    if (algorithm_type == "Regressor") and (algorithm in ["Decision Tree","Random Forest", "Linear Regression", "Ridge Regression", "Gradient Boosting",'Bayesian Ridge Regression','Lasso Regression','Neural Network']): ####----Changed----
        params = add_parameter_regressor(algorithm)
    else :
        params = add_parameter_classifier_general(algorithm)

    # Function to explain overfitting in simple terms
    def explain_overfitting():
        st.write("""
            **What is Overfitting?**
            Overfitting occurs when a model performs exceptionally well on the training data but poorly on the testing data. This happens because the model has learned the noise and specific patterns in the training data, rather than generalizing well to unseen data. 
            Overfitting can lead to misleadingly high accuracy during training, but poor generalization to new data.
        """)
        st.write("""
            **How does Overfitting impact performance?**
            When a model is overfitted, it may predict accurately on the data it was trained on, but it will likely perform poorly on new, unseen data. This reduces the usefulness of the model in real-world applications.
        """)
        st.write("""
            **How can you address Overfitting?**
            Here are some steps you can take to prevent or reduce overfitting:
            - Adjust the training-testing split ratio to ensure you have enough data for testing.
            - Use regularization techniques to avoid overly complex models.
            - Try cross-validation to better evaluate model performance.
        """)

    #Step6-1
    # Now we will build ML Model for this dataset and calculate accuracy for that for classifier
    def model_classifier(algorithm, params):

        if algorithm == 'KNN':
            return KNeighborsClassifier(n_neighbors=params['K'], weights=params['weights'])

        elif algorithm == 'SVM':
            return SVC(C=params['C'], kernel=params['kernel'])

        elif algorithm == 'Decision Tree':
            return DecisionTreeClassifier(
                criterion=params['criterion'], splitter=params['splitter'],
                random_state=params['random_state'])

        elif algorithm == 'Naive Bayes':
            return GaussianNB()

        elif algorithm == 'Random Forest':
            return RandomForestClassifier(n_estimators=params['n_estimators'],
                                        max_depth=params['max_depth'],
                                        criterion=params['criterion'],
                                        random_state=params['random_state'])


        elif algorithm == 'Logistic Regression':
            return LogisticRegression(fit_intercept=params['fit_intercept'],
                                    penalty=params['penalty'], C=params['C'], n_jobs=params['n_jobs'], solver=params['solver'])
        
        elif algorithm == 'Gradient Boosting':
            return GradientBoostingClassifier(loss = params['loss'],n_estimators=params['n_estimators'],learning_rate=params['learning_rate'])

        elif algorithm == 'Neural Network':
            return MLPClassifier(activation=params['activation'], solver=params['solver'], hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=params['learning_rate_init'])


    #Step6-2
    # Now we will build ML Model for this dataset and calculate accuracy for that for regressor
    def model_regressor(algorithm, params):

        if algorithm == 'KNN':
            return KNeighborsRegressor(n_neighbors=params['K'], weights=params['weights'])

        elif algorithm == 'SVM':
            return SVR(C=params['C'], kernel=params['kernel'])

        elif algorithm == 'Decision Tree':
            return DecisionTreeRegressor(
                criterion=params['criterion'], splitter=params['splitter'],
                random_state=params['random_state'])

        elif algorithm == 'Random Forest':
            return RandomForestRegressor(n_estimators=params['n_estimators'],
                                        max_depth=params['max_depth'],
                                        criterion=params['criterion'],
                                        random_state=params['random_state'])

        elif algorithm == 'Linear Regression':
            return LinearRegression(fit_intercept=params['fit_intercept'], n_jobs=params['n_jobs'])
        
        elif algorithm == 'Ridge Regression': ###----Added-----
            return Ridge(alpha=params['alpha'])
        
        elif algorithm == 'Gradient Boosting':
            return GradientBoostingRegressor(loss = params['loss'],n_estimators=params['n_estimators'],learning_rate=params['learning_rate'])
        
        elif algorithm == 'Bayesian Ridge Regression':
            return BayesianRidge(alpha_1=params['alpha_1'], alpha_2=params['alpha_2'], lambda_1=params['lambda_1'], lambda_2=params['lambda_2'])
        
        elif algorithm == 'Lasso Regression':
            return Lasso(alpha=params['alpha'])
    
        elif algorithm == 'Neural Network':
            return MLPRegressor(activation=params['activation'], solver=params['solver'], hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=params['learning_rate_init'])
        

    
            

    if data is not None: 
        X, Y, features_name, predict_name = input_output(data)

        if len(X.columns) > 0 and Y is not None:
            
            # Now selecting classifier or regressor
            # Calling Function based on regressor and classifier
            if algorithm_type == "Regressor":
                model = model_regressor(algorithm,params)
            else :
                model = model_classifier(algorithm,params)

            # Now splitting into Testing and Training data
            split_size = st.slider('Data Split Ratio (% for Training Set)', 10, 90, 80, 5)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=split_size)

            # Detecting Linearly Dependent Features
            st.write("**Regarding Linearly Dependent Features:**")
            st.write("""
                HEART automatically detects linearly dependent features during modeling. It calculates the correlation matrix and uses rank-reduction methods to identify highly correlated features, alerting users and providing recommendations, such as removing affected features.
            """)

            # Calculate the correlation matrix
            correlation_matrix = np.corrcoef(x_train, rowvar=False)
            st.write("**Correlation Matrix:**")
            st.write(correlation_matrix)

            # Identifying highly correlated features
            highly_correlated_features = np.where(np.abs(correlation_matrix) > 0.9)
            highly_correlated_pairs = [(features_name[i], features_name[j]) for i, j in zip(*highly_correlated_features) if i != j]

            if highly_correlated_pairs:
                st.warning("Highly correlated features detected:")
                for pair in highly_correlated_pairs:
                    st.write(f"Feature Pair: {pair}")
                st.write("It is recommended to remove or combine these features to improve model performance.")
            else:
                st.write("No highly correlated features detected.")


            with st.spinner('Model Training...'):
                # Training algorithm
                model.fit(x_train,y_train)


                if algorithm in ['Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Bayesian Ridge Regression']:
                    importance = model.coef_
                    st.write("**Coefficient value for each feature:**")
                    for i,v in enumerate(importance):
                        st.write('Feature: {}, Score: {}'.format(features_name[i], v))
                    fig = plot_chart(features_name, importance)
                    st.pyplot(fig)
                elif algorithm in ['Gradient Boosting']:
                    importance = model.feature_importances_
                    st.write("**Coefficient value for each feature:**")
                    for i,v in enumerate(importance):
                        st.write('Feature: {}, Score: {}'.format(features_name[i], v))
                    fig = plot_chart(features_name, importance)
                    st.pyplot(fig)

                # Now we will find the predicted values
                predict = model.predict(x_test)

                # Finding Accuracy
                if algorithm != 'Linear Regression' and algorithm_type != 'Regressor':
                    # Calculate training and testing accuracy
                    train_accuracy = model.score(x_train, y_train) * 100
                    test_accuracy = accuracy_score(y_test, predict) * 100

                    st.write(f"Training Accuracy is: {train_accuracy:.2f}%")
                    st.write(f"Testing Accuracy is: {test_accuracy:.2f}%")

                    # Check for overfitting: if training accuracy is significantly higher than testing accuracy
                    if train_accuracy - test_accuracy > 10:  # Example threshold for warning
                        st.warning("Warning: The training accuracy is significantly higher than the testing accuracy, which may indicate overfitting.")
                        explain_overfitting()

                    # Additional metrics for binary classification
                    if set(np.unique(Y)) == {0, 1}:  # Checking if the task is binary classification
                        tn, fp, fn, tp = confusion_matrix(y_test, predict).ravel()
                        sensitivity = tp / (tp + fn)
                        specificity = tn / (tn + fp)
                        precision = precision_score(y_test, predict)

                        try:
                            y_scores = model.predict_proba(x_test)[:, 1]
                            auc = roc_auc_score(y_test, y_scores)
                            st.write(f"AUC is: {auc:.2f}")
                        except AttributeError:
                            st.write("AUC calculation requires predict_proba support by the model")

                        st.write(f"Sensitivity is: {sensitivity:.2f}")
                        st.write(f"Specificity is: {specificity:.2f}")
                        st.write(f"Precision is: {precision:.2f}")

                else:
                    # For linear regression, find error metrics
                    st.write(f"Mean Squared Error is: {mean_squared_error(y_test, predict):.2f}")
                    st.write(f"Mean Absolute Error is: {mean_absolute_error(y_test, predict):.2f}")

                ##ADD PREDICTION
                ##MAY ADD VISUALIZATION FOR TUNING

            #Initialise the key in session state
            if 'clicked' not in st.session_state:
                st.session_state.clicked ={1:False}

            #Function to udpate the value in session state
            def clicked(button):
                st.session_state.clicked[button]= True

            st.button("Let's make predictions using this model", on_click = clicked, args=[1])

            if st.session_state.clicked[1]:

                # After training and evaluation, add user input functionality for prediction
                st.write("Input Data for Prediction")
                input_data = {}  # Dictionary to store user inputs
                for feature in X.columns:
                    # You can customize the input method based on the type of data (numeric, categorical, etc.)
                    # Here, we're assuming numeric inputs. For categorical data, consider using st.selectbox or similar.
                    input_data[feature] = st.number_input(f"Input value for {feature}", format="%f")
                
                # Convert user inputs into a DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                prediction = model.predict(input_df)
                
                # Display prediction
                st.write(f"Prediction for {predict_name}: {prediction[0]}")

def plot_chart(features, importance):
    fig, ax = plt.subplots()
    ax.bar(features, importance)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance')
    ax.set_xticks(features)
    ax.set_xticklabels(features, rotation=45, ha='right')
    return fig