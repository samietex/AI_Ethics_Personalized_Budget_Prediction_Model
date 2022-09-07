# AI_Ethics_Personalized_Budget_Prediction_Model
Personalization is a central aspect of many core AI systems. In this project, I will be working on a hypothetical use case for a personalized "activity recommender". The use case has a medium ethical AI risk level and involves a synthetic dataset.

IDOOU is a mobile app users can leverage to get recommendations on activities they can take in a given area, like “visiting a movie theater”, “visiting a park”, “sightseeing”, “hiking”, or “visiting a library”.

## Problem Statement
IDOOU's creators would like to identify if users with bachelor's and master's degrees are a privileged group in terms of budget. In other words, do users with higher education credentials beyond high school have a budget >= $300 compared to users of the app who graduated from high school?

I am tasked with designing IDOOU's newest AI model to predict the budget of its users (in US dollars) given information such as their gender, age, and education_level. I will also explore the provided data and analyze and evaluate this budget predictor's fairness and bias issues.

**Here are the step-by-step approach I took throughout this project:**

* Data exploration by analyzing and evaluating the fairness and bias issues in the data
* Built two machine learning models (Logistic Regression and Gaussian Naive Bayes) and also went ahead to evaluate the performance of the models.
* Performed some analysis and evaluation on the fairness and bias issues in the AI solution.
* The details of the use case was documented in the `model_card`...
* One of the models (Logistic Regression model) was selected...I then went ahead to build an explainable AI strategy of my choice to understand the model's predictions.
* I implemented the Reweighing preprocessing bias mitigation strategy and then evaluated the improvements in fairness on the data.


### Ethical Considerations

* The objective of the app is to remove users from having to handle the nitty-gritty details of 
finding the right activity, like determining the appropriate budget, ensuring the weather is perfect, 
and the location/accommodation is not closed, so users don't have the liberty to control most aspects 
of the model. 
* The model for this app does not have any implications on human life, health or safety in the usage
of the model.
* The key contributing factors from the permutation importance after the preprocessing bias mitigation
are users in all the age groups(18-24, 25-44, 45-65, 66-92), and users in the bachelor's and master's 
education level.

### Caveats and Recommendations

* I'll recommend Logistic Regression model for classification related problems in order to have a better
accuracy score and better fairness metric scores.
* Preprocessing and postprocesssing bias mitigation techniques should also be applied on the dataset
of the model.
* The Logistic Regression model after the preprocessing bias mitigation seems to have higher false positives and 
lesser false negatives compared to the Logistic Regression before the bias mitigation.



