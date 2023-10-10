## Model

We conducted experiments using two different machine learning models to predict our target variable. The models we tried are logistic regression and XGBoost. Below, we present the evaluation metrics for both models:

### XGBoost

- **Precision:** 0.88 (Class 0), 0.25 (Class 1)
- **Recall:** 0.52 (Class 0), 0.69 (Class 1)
- **F1-Score:** 0.66 (Class 0), 0.37 (Class 1)
- **Accuracy:** 0.55
- **Macro Avg Precision:** 0.56
- **Macro Avg Recall:** 0.61
- **Macro Avg F1-Score:** 0.51
- **Weighted Avg Precision:** 0.76
- **Weighted Avg Recall:** 0.55
- **Weighted Avg F1-Score:** 0.60
- **Support:** Class 0: 18294, Class 1: 4214

### Logistic Regression

- **Precision:** 0.88 (Class 0), 0.25 (Class 1)
- **Recall:** 0.52 (Class 0), 0.69 (Class 1)
- **F1-Score:** 0.65 (Class 0), 0.36 (Class 1)
- **Accuracy:** 0.55
- **Macro Avg Precision:** 0.56
- **Macro Avg Recall:** 0.60
- **Macro Avg F1-Score:** 0.51
- **Weighted Avg Precision:** 0.76
- **Weighted Avg Recall:** 0.55
- **Weighted Avg F1-Score:** 0.60
- **Support:** Class 0: 18294, Class 1: 4214

Both models achieved similar accuracy of 0.55. However, when we focus on the F1-score, which balances precision and recall, we can see that XGBoost has a higher F1-score for both Class 0 (0.66) and Class 1 (0.37) compared to logistic regression (Class 0: 0.65, Class 1: 0.36).

Due to the higher F1-scores across both classes, we have chosen XGBoost as our final model for this task. The higher F1-scores indicate that XGBoost strikes a better balance between correctly classifying instances of both classes, making it a more suitable choice for our problem.

## FastAPI for Model Deployment

To operationalize our machine learning model and make real-time predictions, we have developed a FastAPI application. FastAPI is a modern web framework for building APIs quickly and efficiently.

After training, we have saved the trained model to a file named "model.joblib." This serialized model file can be easily loaded and used for making predictions in our FastAPI application.

In the FastAPI application, we have created an endpoint that consumes generated model and uses it to make predictions based on incoming data. Users can send requests to this endpoint, and the model will return predictions promptly.

## Continuous Integration and Continuous Deployment (CI/CD)

We have implemented a robust CI/CD pipeline to automate the deployment process of our FastAPI application to Google App Engine. This ensures that code changes are thoroughly tested and seamlessly deployed to our production environment.

In our CI pipeline, which is defined in the `ci.yaml` file, we have included a comprehensive set of unit tests that are automatically triggered whenever a pull request is made to the `master` branch. By running tests in an automated fashion, we ensure that only well-tested code is merged into our main branch.

Our CD (Continuous Deployment) pipeline, defined in the `cd.yaml` file, handles the deployment of our FastAPI application to Google App Engine. Once the code changes have passed all tests in the CI pipeline and are merged into the `master` branch, the CD pipeline automatically deploys the updated application to the production environment.

Our deployed application can be accessed using the following URL: [https://retropixel-396819.uc.r.appspot.com](https://retropixel-396819.uc.r.appspot.com). This link provides access to our live FastAPI application, allowing users to interact with the prediction model in real-time.

## Makefile for Streamlined Development

To facilitate development, testing, and deployment processes, we've created a Makefile within our project. The Makefile serves as a versatile tool for running various tasks and commands efficiently.

The Makefile includes targets for running unit tests related to our FastAPI application. These unit tests cover different aspects of our project, including the behavior of our API and the functionality of the model class. By using coroutines, we ensure that our tests run efficiently and can be easily integrated into our CI/CD pipelines.