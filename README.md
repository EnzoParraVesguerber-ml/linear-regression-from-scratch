# 📊 Linear Regression on Advertising Dataset

This project explores **Linear Regression** techniques applied to the [Advertising dataset](https://www.statlearning.com/), with the objective of predicting product **Sales** based on marketing investments in **Radio** and **Newspaper**.

The project goes beyond basic regression — we implement the entire modeling pipeline **from scratch** using NumPy and other scientific libraries, with thorough statistical analysis and model diagnostics.

---

## 🔍 Key Features

✅ **Implemented Linear Regression manually** using both:
- The **Normal Equation** method
- The **QR decomposition** method (for improved numerical stability)

✅ **Built two regression models**:
- A **Base Model** with `Radio` and `Newspaper` as predictors
- An **Interaction Model** with an added term: `Radio × Newspaper`

✅ **Computed Confidence Intervals (CIs)** for all model coefficients:
- Manually derived from the variance-covariance matrix
- Validated using standard error propagation and Student's t-distribution

✅ **Checked for multicollinearity** using **VIF (Variance Inflation Factor)**:
- Implemented from scratch
- Ensured that predictors were not highly correlated

✅ **Thorough documentation and educational purpose**:
- Clean and readable code with comments in English
- Step-by-step explanations in markdown cells


---

## 📊 Model Comparisons

| Feature               | Base Model | Interaction Model |
|----------------------|------------|-------------------|
| Predictors           | Radio, Newspaper | Radio, Newspaper, Radio × Newspaper |
| Method               | Normal Equation | Same |
| VIF Checked          | ✅          | ✅                |
| Confidence Intervals | ✅          | ✅                |

By comparing both models, we were able to investigate:
- Whether interaction between variables improves performance
- The impact of adding potentially redundant predictors
- The precision of coefficient estimates

---

## 📈 Statistical Outputs

The notebook includes:
- Coefficients and standard errors
- 95% Confidence intervals
- VIF values per predictor
- Model weights via QR and Normal Equation (for comparison)

These outputs provide statistical insight into the relationships between features and the target variable, as well as the robustness of the model under different linear solving techniques.

---







