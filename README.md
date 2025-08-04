# PREDICTING-ELIGIBILITY-FOR-USING-MACHINE-LEARNING.




                                CAPSTONE PROJECT

                       PREDICTING ELIGIBILITY FOR USING MACHINE LEARNING.

Presented By:
1.	PANDI SRIRAM
2.	STUDENT ID: STU6832ecf599a031748167925
3.	AURORA PG COLLEGE(RAMANTHAPUR)-MCA
 
OUTLINE
	Problem Statement (Should not include solution)
	Proposed System/Solution
	System Development Approach (Technology Used)
	Algorithm & Deployment
	Result (Output Image)
	Conclusion
	Future Scope
	References
 

PROBLEM STATEMENT

The National Social Assistance Program (NSAP) offers critical financial aid to the elderly, widows, and persons with disabilities from below-poverty-line (BPL) households through various schemes. However, identifying the right beneficiaries for each sub-scheme is often a manual, time-consuming, and error-prone task, which can lead to delays or incorrect scheme allocation. This affects the timely disbursement of aid and the overall efficiency of the welfare program. There is a need for an intelligent system that can assist in automating the classification of applicants into the most appropriate NSAP scheme based on available demographic and socio-economic data. 

PROPOSED SOLUTION


 
We propose a machine learning-based multi-class classification system that predicts the appropriate NSAP scheme for a given applicant. By leveraging the AI Kosh dataset, the system will learn patterns from historical data to automate and improve the decision-making process for scheme assignment. This tool will assist government agencies in reducing manual workload, minimizing errors, and ensuring faster and more accurate allocation of welfare benefits. 

SYSTEM APPROACH

1)Programming Language: Python

2)Libraries/Frameworks:
•	Data Analysis: pandas, numpy
•	Data Visualization: matplotlib, seaborn
•	Machine Learning: scikit-learn, xgboost, lightgbm
•	Model Evaluation: classification_report, confusion_matrix, cross_val_score

3)IDE/Environment: Jupyter Notebook / Google Colab

4)Deployment (Optional): Streamlit / Flask for Web Interface
 

ALGORITHM & DEPLOYMENT



Algorithms Used:
•	Logistic Regression (Baseline)
•	Random Forest
•	XGBoost (Preferred for handling imbalanced multi-class datasets)
•	LightGBM (Alternative to XGBoost for faster training on large datasets)


Workflow:
1.	Data Collection: Use the AI Kosh dataset.
2.	Data Preprocessing: Handle missing values, encode categorical data, normalize features.
3.	Feature Engineering: Create meaningful variables based on age, gender, income, disability status, etc.
4.	Model Training: Train multiple classifiers and fine-tune hyperparameters.
5.	Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
6.	Deployment (Optional): Wrap the model using Streamlit or Flask for real time predictions.
 

RESULT
<img width="1875" height="662" alt="image" src="https://github.com/user-attachments/assets/83544a3d-7264-43d6-a002-8bc759da727e" />
<img width="1875" height="803" alt="image" src="https://github.com/user-attachments/assets/ad30c5f6-22be-440f-90fd-af2ac66d5c58" />
<img width="1553" height="694" alt="image" src="https://github.com/user-attachments/assets/a3610fa2-3ead-49c4-b9d6-2917d540cfce" />
<img width="1875" height="859" alt="image" src="https://github.com/user-attachments/assets/56d7cd7b-c68f-4216-a759-ce5e75d8c898" />
<img width="1429" height="679" alt="image" src="https://github.com/user-attachments/assets/98ee7bd3-96dc-4999-b2d9-b82abae2c7b6" />
<img width="1875" height="903" alt="image" src="https://github.com/user-attachments/assets/98b183db-d579-489c-95ae-847ce9ec5002" />
<img width="1875" height="888" alt="image" src="https://github.com/user-attachments/assets/8a6dd6a5-b855-4fcc-87f0-c8bd1cf37080" />
<img width="1781" height="825" alt="image" src="https://github.com/user-attachments/assets/bff5482d-9495-4a8c-9d40-5dd5b63352c7" />
<img width="1875" height="838" alt="image" src="https://github.com/user-attachments/assets/3e0a853c-b00d-4a1a-8673-fc86c5b87308" />

  
 


 


 

 

 
 
 

 


CONCLUSION


The machine learning-based approach for NSAP scheme prediction significantly improves the accuracy and efficiency of scheme allocation. With automated eligibility prediction, the system can reduce manual errors and speed up the distribution process. Among all algorithms tested, XGBoost provided the most balanced performance in terms of accuracy and generalization.
 


FUTURE SCOPE



1)Integrate real-time data collection from government databases.
2)Expand the model to include other social welfare schemes.
3)Deploy the model as a mobile application for local governance use.
4)Implement explainable AI (XAI) techniques for better transparency in predictions.
5)Use NLP for processing unstructured text data from applications.
 
REFERENCES

1) AI Kosh NSAP Dataset
2) Scikit-learn Documentation: https://scikit-learn.org/
3) XGBoost Documentation: https://xgboost.readthedocs.io/
4) LightGBM Documentation: https://lightgbm.readthedocs.io/
5) Government of India NSAP Portal: https://nsap.nic.in/
 
     
 
             		  
	  





















THANK YOU
