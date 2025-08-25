# ⚽ Beat the Bookie  

> Using Machine Learning to Predict English Premier League (EPL) Football Match Outcomes  

---

## 📖 Overview  
This project explores the use of **machine learning models** to predict the outcomes of EPL football matches:  
**Home Win, Away Win, or Draw**.  

We combined **match statistics**, **engineered features**, and **external factors** (such as fatigue, spending, and distance traveled) to develop predictive models that outperform random guessing.  

Our best-performing model achieved a **testing accuracy of 66.5%** using **Linear Discriminant Analysis (LDA)**.  

---

## 🏗️ Features & Data  
We engineered features from a variety of sources to capture match dynamics and external influences:  
- ⚡ **Expected Match Value (xM)** – derived from shot-based and non-shot-based regression models (inspired by Expected Goals, xG).  
- 📈 **ELO Rating** – adapted from sports analytics to measure relative team strength.  
- 💰 **Spending & Net Transfers** – normalized season-by-season to reflect squad strength.  
- 💤 **Fatigue** – modeled using a sigmoid decay based on days of rest.  
- 🚍 **Distance Travelled (Away Teams)** – a proxy for home advantage.  
- 🔥 **Recent Form (Last N Games)** – optimized at N = 5 matches.  

---

## 📊 Results  
- **Best Regression Model (Shot-based):** Support Vector Machine  
- **Best Regression Model (Non-shot-based):** Linear Regression  
- **Best Classification Model:** Linear Discriminant Analysis (LDA)  
- **Performance:**  
  - Training Accuracy: **84.6%**  
  - Testing Accuracy: **66.5%**  
