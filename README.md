# 🏨 A Revenue Decision Support Tool for Hotels

## Project Overview

Hotel cancellations create uncertainty for both revenue and operations.
This project explores hotel booking data to identify the main patterns
associated with cancellations and uses Machine Learning to estimate the
cancellation probability of an individual booking.

The final result is a **decision-support tool for hotel managers**,
combining:

-   Exploratory Data Analysis (EDA)
-   Machine Learning
-   An interactive Tableau dashboard
-   A Streamlit prediction application
-   Business-oriented recommendations

The goal is not only to predict cancellation risk, but to turn that
prediction into information that can support better hotel decisions.

------------------------------------------------------------------------

## 🎯 Business Problem

Hotel managers need to identify risky reservations early enough to take
preventive action.

This project addresses two main questions:

1.  **Can we identify cancellation risk before it happens?**
2.  **Can we translate that prediction into useful actions for hotel
    teams?**

------------------------------------------------------------------------

## 📊 Dataset

The project uses historical booking data from **two hotels in Portugal**
--- a City Hotel and a Resort Hotel --- with reservations due to arrive
between **July 2015 and August 2017**.

The data includes booking dates and lead time, guest characteristics,
booking channel, previous booking behaviour, ADR, special requests,
parking requirements, and commercial information.

After data cleaning, the analysis contains approximately **119,000
bookings**.

------------------------------------------------------------------------

## 🔎 Exploratory Data Analysis

The EDA was used to understand which booking characteristics were
associated with higher or lower cancellation rates.

Some of the main patterns identified were:

-   **Longer lead times** were associated with higher cancellation
    rates.
-   **Group bookings** showed a particularly high cancellation rate.
-   **Repeated guests** cancelled less frequently than new guests.
-   Bookings with **no special requests** showed a higher cancellation
    rate than bookings with several special requests.
-   Previous booking behaviour also provided useful information about
    cancellation risk.

An interactive **Tableau dashboard** was created to make these patterns
easier to explore.

> 🔗 **Tableau Dashboard:** https://public.tableau.com/app/profile/elena.garrigues.pascual/viz/Tableau_17860249047140/BookingCancellationDashboard

------------------------------------------------------------------------

## 🤖 Machine Learning

The problem was approached as a **binary classification task**:

-   `0` → Not Cancelled
-   `1` → Cancelled

Two models were compared:

  -----------------------------------------------------------------------------
  Model            Accuracy    Precision       Recall     F1 Score      ROC-AUC
  ------------ ------------ ------------ ------------ ------------ ------------
  Logistic            0.805        0.777        0.664        0.716        0.878
  Regression                                                       

  **Random        **0.884**    **0.881**    **0.796**    **0.836**    **0.953**
  Forest**                                                         
  -----------------------------------------------------------------------------

The **Random Forest** was selected as the final model because it
provided the strongest overall performance.

Its ROC-AUC of **0.953** shows a strong ability to distinguish between
cancelled and non-cancelled bookings.

------------------------------------------------------------------------

## 🧹 Feature Selection

Not every variable explored during the EDA was kept in the final model.

Several features were removed after considering their usefulness,
interpretation and impact on model performance. This helped simplify the
final application while maintaining very similar predictive performance.

One important example was **Deposit Type**. Although it showed a very
strong relationship with cancellations during the EDA, its business
interpretation in the source data was ambiguous. The feature was
therefore excluded from the final model with minimal impact on
performance.

This distinction is important: a variable can be interesting for
exploratory analysis without necessarily being appropriate for the final
predictive tool.

------------------------------------------------------------------------

## 💻 Streamlit Application

The trained Random Forest model was integrated into a **Streamlit
application** called the **Booking Risk Analyzer**.

The application returns:

-   The **estimated cancellation probability**
-   🟢 Low Risk: below 30%
-   🟡 Medium Risk: 30%--70%
-   🔴 High Risk: 70% or above
-   Relevant **risk indicators**
-   A **booking follow-up recommendation**
-   A loyalty recommendation for repeated guests

The application is designed as a **decision-support tool**, not as a
replacement for professional judgement.

> 🔗 **Live Streamlit App:** https://arevenuesupporttool-yzlrswmnfneauo7fq4slec.streamlit.app/

------------------------------------------------------------------------

## 💡 Business Value

The project moves beyond model performance by connecting predictions
with possible hotel actions. It can help hotel teams prioritise
follow-up for higher-risk bookings, identify relevant risk indicators,
adapt communication for repeated guests, and support earlier operational
decisions.

------------------------------------------------------------------------

## ⚠️ Limitations

-   Historical data from **2015--2017**
-   Only **two hotels in Portugal**
-   Some variables have **ambiguous business definitions**
-   The model should be validated with more recent and representative
    hotel data before real-world deployment

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Further **model optimisation**
-   Optimising the decision threshold according to business objectives
-   Validating the model with more recent booking data
-   Moving **from individual to group predictions** by estimating
    expected cancellations for a specific period
-   Using expected cancellations to support occupancy, pricing and
    capacity decisions

------------------------------------------------------------------------

## 🛠️ Technologies

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Streamlit
-   Tableau
-   Jupyter Notebook

------------------------------------------------------------------------

## 📁 Repository Structure

``` text
.
├── .streamlit/
├── models/
│   └── random_forest_pipeline.pkl
├── A Revenue Decision Support Tool.ipynb
├── app.py
├── clean_hotel_booking.csv
├── hotel_bookings.csv
├── Tableau.twb
├── requirements.txt
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

## 📌 Final Takeaway

**Can we identify cancellation risk before it happens?**

**Yes.**

The project shows how hotel booking data can be transformed from
historical information into a practical decision-support tool --- moving
from **predicting cancellations to supporting better hotel decisions**.
