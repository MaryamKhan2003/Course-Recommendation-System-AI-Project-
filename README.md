#  AI-POWERED COURSE RECOMMENDATION SYSTEM

---

##  **PROJECT OVERVIEW**
The **AI-Powered Course Recommendation System** is designed to help students select the most suitable online courses based on their **interests, skill level, and learning preferences**.  

With the rapid growth of online learning platforms, students often face difficulty in choosing the right course. This system solves that problem by providing **personalized recommendations using Machine Learning**.

---

##  **PROBLEM STATEMENT**
Students struggle to:
-  Choose the right course from thousands of options  
-  Match their skill level with course difficulty  
-  Find courses aligned with their interests  
-  Filter courses based on ratings and practical content  

---

##  **PROPOSED SOLUTION**
This project implements a **Machine Learning-based recommendation system** using:

-  **K-Nearest Neighbors (KNN) Algorithm**  
-  **Feature-based similarity matching**  
-  **User preference analysis**  

###  **System Functionality**
- Takes user input (skills, experience level, preferences)  
- Converts input into feature vectors  
- Calculates similarity using Euclidean distance  
- Recommends **Top 10 relevant courses**  
- Predicts **course difficulty level**  

---

##  **MACHINE LEARNING MODEL**

###  **Algorithm Used:**
**K-Nearest Neighbors (KNN)**

###  **Why KNN?**
- ✔ Simple and intuitive  
- ✔ No training phase required (lazy learning)  
- ✔ Ideal for similarity-based recommendations  
- ✔ Easily adaptable with new data  

---

##  **DATASET DESCRIPTION**

###  **Source:**
Kaggle (Coursera Courses Dataset)

###  **Dataset Statistics:**
-  **Total Courses:** 500+  
-  **Original Features:** 7  
-  **Processed Features:** 16  

###  **Key Attributes:**
- Course Name  
- University  
- Difficulty Level  
- Course Rating  
- Course Description  
- Skills  

---

##  **TECHNOLOGIES USED**

| **Technology** | **Purpose** |
|--------------|------------|
| **Python** | Core programming language |
| **Flask** | Web application framework |
| **Pandas** | Data processing |
| **NumPy** | Mathematical computations |
| **Matplotlib** | Data visualization |
| **HTML/CSS/JS** | Frontend development |

---

##  **SYSTEM WORKFLOW**

1.  User enters preferences (skills, level, interests)  
2.  Data is converted into feature vector  
3.  KNN computes similarity (Euclidean distance)  
4.  Nearest courses are identified  
5. Top 10 courses are recommended  

---

##  **MODEL PERFORMANCE**

-  **Accuracy:** ~86.7%  
- **Validation:** 5-Fold Cross Validation  
-  **Variance:** Low (±1%)  

---

##  **SYSTEM INTERFACE**

-  **Input Page:** User enters profile details  
-  **Statistics Page:** Model performance graphs  
-  **Results Page:** Recommended courses displayed  

---

## **PROJECT STRUCTURE**

Course-Recommendation-System/
│── app.py
│──recommender.py
│── data/
  ├──coursera_courses.csv
│── templates/
│ ├── index.html
│ ├── stats.html
│── requirement.txt

## **CONTRIBUTORS
Rimsha Farid
Khadeeja Hafeez
Maryam Khan
