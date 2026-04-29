# 🚗 Vehicle Breakdown Prediction - Flask ML App

A machine learning-based web application that predicts the likelihood of vehicle breakdown using real-time input parameters. This system helps in **early maintenance decisions** and reduces unexpected failures.

---

## 📌 Project Overview

We developed a **machine learning-based approach** to predict vehicle breakdowns using an **ensemble learning technique**.

The model uses a **Voting Classifier** that combines:
- Logistic Regression  
- Naïve Bayes  
- Support Vector Machine (SVM)  

This ensemble improves prediction accuracy by leveraging the strengths of multiple models.

---

## 📊 Dataset Details

- Total Records: **110,446**
- Source: Vehicle engine sensor data  
- Total Features: **10**

### 🔢 Features Used

1. Engine Coolant Temperature (°C)  
2. Ambient Air Temperature (°C)  
3. Accelerator Pedal Position D (%)  
4. Intake Manifold Absolute Pressure (kPa)  
5. Engine Rotation Speed (RPM)  
6. Vehicle Speed (km/h)  
7. Intake Air Temperature (°C)  
8. Mass Air Flow Sensor (g/s)  
9. Absolute Throttle Position (%)  
10. Accelerator Pedal Position E (%)  

---

## 🧠 Model Performance Summary

| Model                  | Accuracy |
|-----------------------|----------|
| Logistic Regression   | 0.89     |
| Support Vector Machine| 0.91     |
| Naïve Bayes           | 0.88     |
| Voting Classifier     | **0.94** |

✅ The **Voting Classifier** achieved the best performance.

---

## 🚀 Features

1. Input vehicle parameters to predict breakdown probability  
2. Simple and user-friendly web interface  
3. Pre-trained ML model for real-time predictions  
4. Ensemble learning for improved accuracy  

---

## 🌐 Deployed Project  

👉 https://flask-ml-app-cve6.onrender.com/

---

## 🛠️ Technologies Used  

- **Flask** – Web framework  
- **Scikit-learn** – Machine learning models  
- **HTML/CSS** – Frontend design  
- **Python** – Backend logic  

---

## ⚙️ How It Works  

1. User enters vehicle sensor data  
2. Data is preprocessed and fed into the model  
3. Ensemble model predicts breakdown likelihood  
4. Result is displayed instantly  

---

## 📌 Future Improvements  

- Add real-time sensor integration  
- Improve UI/UX design  
- Deploy as a mobile application  
- Add alert system for high-risk predictions  

---

## 👨‍💻 Author  

**Gaurav Saklani**

---

## 📄 License  

MIT License
