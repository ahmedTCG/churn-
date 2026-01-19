## Churn Model — Summary

**Goal**  
Predict customer churn, defined as **no strong activity in the next 30 days**, using historical interaction data.

**Data**  
- 1 year of customer interactions (Jan 2025 – Jan 2026)  
- ~4.7M events, ~70k customers per snapshot  

**Model & Features**  
- Customer-level features: recency, frequency, active days, event-type mix, order behavior  
- Regularized logistic model with probability calibration  
- Identical feature logic used for training, evaluation, and inference  

**Validation (Key Result)**  
- **Time-based validation (no leakage)** using monthly snapshots  
- Model trained on earlier months and evaluated on future months  

**Performance (future months)**  
- **ROC-AUC ≈ 0.89**  
- **PR-AUC ≈ 0.89**  
- Metrics are stable across months  

**Conclusion**  
The model generalizes well to future data and shows no evidence of temporal leakage (lookahead bias) or temporal overfitting.  
It is suitable for production or business decision-making.

**Next Steps (Optional)**  
- Define business thresholds (top-K or precision-based)  
- Monitor score and churn-rate stability over time  
- Adjust retraining cadence as needed  
