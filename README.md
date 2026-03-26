Brugada Syndrome Detection using ECG Signals

This project aims to detect Brugada Syndrome from ECG signals using machine learning. It processes multi-lead ECG recordings, extracts statistical features, and trains a classification model to distinguish between normal and Brugada cases.


**Features**

* Load ECG metadata and labels from CSV
* Read ECG signal data using WFDB format
* Extract statistical features from multiple ECG leads:

  * Mean
  * Standard deviation
  * Maximum value
  * Minimum value
* Train a machine learning model (**Random Forest Classifier**)
* Evaluate performance using accuracy and classification report
* Predict and visualize ECG for a specific patient
  

**Project Structure**

project/
│
├── brugadaProject.py       # Main script
├── metadata.csv            # Patient labels (not included if sensitive)
├── data/                   # ECG signal files (WFDB format)
├── .gitignore
└── README.md


**Dataset**

* Metadata file: `metadata.csv`
  * Contains:
    * `patient_id`
    * `brugada` (label: 0 = normal, 1 = Brugada)
      
* ECG signals:
  * Stored per patient using WFDB format
  * Loaded via patient ID



**Run by:**

1. Installing dependencies
pip install numpy pandas matplotlib scikit-learn wfdb

2. Set dataset path
base_path = "C:/Users/user/PyCharmMiscProject/brugada-huca/files"

3. Run the script
python brugadaProject.py


**Machine Learning Model**

* Algorithm: Random Forest Classifier
* Parameters:
  * `n_estimators = 300`
  * `class_weight = balanced`
* Data split:
  * 80% training
  * 20% testing
* Features are normalized using StandardScaler


**Output**
* Dataset statistics
* Model accuracy
* Classification report:
  * Precision
  * Recall
  * F1-score


**Prediction Example**

The model can predict for a specific patient:
predict_patient("188981")

Output:

* ECG plot (Lead V1)
* Prediction result (Brugada / Normal)
* Confidence score

**Limitations of this project**

* Uses simple statistical features (not deep learning)


**Team IDSC**
1. Nur Dalili binti Kamal
2. Nurul Afiqah binti Yunos
3. Prithika Latchumy a/p Manivel
4. Saoumya a/p Arjuna


**This project is for educational and research purposes only**
