# Vehicle Sales Forecasting: Time Series Analysis with Economic Indicators

A comprehensive time series forecasting project comparing SARIMA, SARIMAX, Prophet, ETS, XGBoost, and ensemble methods to predict vehicle sales using unemployment rate data.

## 📊 Project Overview

This project investigates the impact of the Maryland Unemployment Rate (MDUR) on forecasting new vehicle sales from 2002 to January 2025. Seven different forecasting models were implemented and compared to determine the optimal approach for accurate sales prediction.

**Key Achievement:** The weighted ensemble model achieved a **MAPE of 6.58%**, outperforming the baseline SARIMA model (7.63% MAPE) by over 14%.

## 🎯 Objectives

- Evaluate the Maryland Unemployment Rate's contribution to forecasting accuracy
- Compare multiple time series and machine learning forecasting techniques
- Develop an optimized ensemble model for vehicle sales prediction
- Provide insights for future forecasting improvements

## 📁 Dataset

**Time Period:** Monthly data from 2002 to January 2025
- **Training Period:** 2002–2022
- **Test Period:** 2023–2025 (25 months)

**Data Sources:**
- **Vehicle Sales Data:** Maryland Department of Transportation - Motor Vehicle Administration
- **Unemployment Data:** Federal Reserve Economic Data (FRED) - Maryland Unemployment Rate (MDUR)

**Features:**
- Monthly new vehicle sales
- Maryland Unemployment Rate (standardized via z-score normalization)
- Time index and month dummy variables (for XGBoost)

## 🔧 Models Implemented

### 1. SARIMA (Baseline)
- Seasonal AutoRegressive Integrated Moving Average
- Auto ARIMA with 12-month seasonal period
- Constrained parameters for stability
- **MAPE: 7.63%**

### 2. SARIMAX
- Extended SARIMA with MDUR as exogenous variable
- Reduced seasonal complexity
- **MAPE: 7.77%**

### 3. Prophet (with and without MDUR)
- Facebook's forecasting tool
- Custom monthly seasonality configuration
- Additive seasonality mode
- **MAPE with MDUR: 11.06%**
- **MAPE without MDUR: 8.16%**

### 4. ETS with MDUR
- Error-Trend-Seasonality model
- Hybrid approach: detrending with MDUR, then ETS on residuals
- Additive error, trend, and seasonality
- **MAPE: 8.56%**

### 5. XGBoost
- Gradient boosting with MDUR, time index, and month dummies
- Fixed hyperparameters
- **MAPE: 10.06%**

### 6. Weighted Ensemble (Winner 🏆)
- Combined SARIMA, ETS with MDUR, and Prophet with MDUR
- Inverse-MAPE weighting scheme
- **MAPE: 6.58%** ✨

## 📈 Key Results

| Model | MAPE | Performance vs Baseline |
|-------|------|------------------------|
| **Weighted Ensemble** | **6.58%** | **+14% improvement** |
| SARIMA (Baseline) | 7.63% | — |
| SARIMAX | 7.77% | -2% |
| Prophet (no MDUR) | 8.16% | -7% |
| ETS with MDUR | 8.56% | -12% |
| XGBoost | 10.06% | -32% |
| Prophet with MDUR | 11.06% | -45% |

### Impact of Unemployment Rate

The Maryland Unemployment Rate showed **mixed effects** across different models:

✅ **Positive Impact:**
- Ensemble model: Significant improvement in accuracy
- Captured economic trends affecting consumer purchasing behavior

⚠️ **Inconsistent Impact:**
- SARIMAX: Minimal improvement (linear relationship assumption may be limiting)
- Prophet: Performance degraded with MDUR (regressor handling issues)
- ETS: Moderate utility
- XGBoost: Shows potential but needs more feature engineering

**Key Insight:** Economic indicators enhance forecasting in ensemble frameworks, but their integration requires careful model-specific tuning.

## 🛠️ Technical Stack

- **Language:** Python 3.x
- **Time Series Models:** `statsmodels` (SARIMA/SARIMAX), `statsmodels.tsa.holtwinters` (ETS)
- **Modern Forecasting:** `Prophet`
- **Machine Learning:** `XGBoost`
- **Data Processing:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`

**Evaluation Metrics:**
- Primary: MAPE (Mean Absolute Percentage Error)
- Supporting: MAE, MSE, RMSE, MAD

## 📊 Visualizations

The project includes comprehensive visual analysis:

- **Time Series Overview:** Vehicle sales with 12-month moving average showing long-term trends
- **Decomposition Analysis:** Trend, seasonality, and residual components
- **Model Comparison:** Actual vs. forecasted sales for all models (2023–2025)
- **Error Analysis:** Forecast errors and residuals for each model
- **Ensemble Performance:** Detailed analysis of the winning model's predictions

## 💡 Key Findings

1. **Ensemble methods effectively combine diverse modeling strengths** to achieve superior accuracy
2. **Economic indicators enhance predictions** when properly integrated, especially in ensemble frameworks
3. **Individual model performance varies significantly** with exogenous variables, suggesting the need for model-specific optimization
4. **Prophet requires careful regressor handling**, potentially benefiting from multiplicative seasonality
5. **XGBoost shows promise** with expanded feature engineering (lagged variables, interactions)

## 🚀 Future Improvements

- **Lagged Variables:** Incorporate time-delayed unemployment effects (consumer behavior lag)
- **Additional Economic Indicators:** Interest rates, consumer confidence index, gas prices
- **Advanced Feature Engineering:** Interaction terms, rolling statistics, holiday effects
- **Refined Prophet Configuration:** Test multiplicative seasonality for better regressor integration
- **Enhanced Ensemble Methods:** Explore stacking, boosting, or neural network ensembles
- **Hyperparameter Optimization:** Grid search or Bayesian optimization for XGBoost
- **Regional Expansion:** Extend analysis to multiple states or national-level data

## 📚 References

- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.)
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system
- Zhang, Y., & Zhong, M. (2017). Forecasting electric vehicles sales with univariate and multivariate time series models
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 competition: 100,000 time series and 61 forecasting methods

## 📝 Usage

```python
# Clone the repository
git clone https://github.com/yourusername/vehicle-sales-forecasting.git

# Install dependencies
pip install -r requirements.txt

# Run the main analysis
python forecasting_analysis.py

# Or explore the Jupyter notebook
jupyter notebook vehicle_sales_forecasting.ipynb
```

## 🤝 Contributing

Feel free to fork this repository and submit pull requests. Suggestions for improvements and extensions are welcome!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Sivaram Senthilkumar**
- Department of Mechanical and Industrial Engineering
- University of Massachusetts Amherst
- Course: 622 - Predictive Analytics and Statistical Learning

---

⭐ If you find this project helpful, please consider giving it a star!

📧 Questions or feedback? Feel free to open an issue or reach out.
