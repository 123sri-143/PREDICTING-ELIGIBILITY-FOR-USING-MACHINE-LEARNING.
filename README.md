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
<img width="1469" height="618" alt="image" src="https://github.com/user-attachments/assets/eff226fe-f660-4e65-9355-ec2099204a16" />
<img width="1874" height="802" alt="image" src="https://github.com/user-attachments/assets/9c7a7af6-13e6-4566-8db1-cc50b1c33cbd" />
<img width="1902" height="828" alt="image" src="https://github.com/user-attachments/assets/eb43cb10-5746-4103-a609-8ad298f621c7" />
<img width="1941" height="650" alt="image" src="https://github.com/user-attachments/assets/a6f476a8-26fa-465f-8368-90a781dfbe62" />
<img width="1905" height="503" alt="image" src="https://github.com/user-attachments/assets/09706111-93a3-48b3-8379-b09ab90b2f9f" />
<img width="1917" height="897" alt="image" src="https://github.com/user-attachments/assets/37eab1c6-2232-462c-96f1-b3c7eb0c6098" />
<img width="1941" height="810" alt="image" src="https://github.com/user-attachments/assets/59d4e326-7258-45a6-8710-501b9793b331" />
<img width="1905" height="249" alt="image" src="https://github.com/user-attachments/assets/6d451af7-b83b-420f-aa52-cd198a17e55d" />
<img width="1941" height="947" alt="image" src="https://github.com/user-attachments/assets/a7980631-46d5-40e2-bdb7-12f5c311429b" />
<img width="1941" height="943" alt="image" src="https://github.com/user-attachments/assets/2efe372f-ce89-4900-84ae-e96ab54417d3" />
<img width="1788" height="969" alt="image" src="https://github.com/user-attachments/assets/27b65b54-3fa5-49f9-8dce-af0949e9c98e" />
<img width="1919" height="832" alt="image" src="https://github.com/user-attachments/assets/9b25eff9-05dd-46fa-8efd-20af25a33568" />
<img width="1941" height="682" alt="image" src="https://github.com/user-attachments/assets/7643c4b0-b0f1-4648-92bd-b534e0173161" />
<img width="1941" height="927" alt="image" src="https://github.com/user-attachments/assets/f42950c6-fbdf-45ea-8155-1463f62d0382" />
<img width="1912" height="853" alt="image" src="https://github.com/user-attachments/assets/acd59e92-0878-41ee-b331-401e33bedb29" />


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
 
 IBM CERTIFICATIONS    
<img width="1152" height="864" alt="image" src="https://github.com/user-attachments/assets/3d08019f-2480-47b6-82f2-b0ede839904a" />
<img width="1158" height="865" alt="image" src="https://github.com/user-attachments/assets/72f4621a-4282-4526-b333-7a87171f18f6" />
<img width="1337" height="865" alt="image" src="https://github.com/user-attachments/assets/22f069a7-d0c2-48d1-a7b8-b88075420a98" />

THANK YOU
