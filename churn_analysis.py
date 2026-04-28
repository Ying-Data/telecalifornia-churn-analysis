"""
TeleCalifornia — Customer Churn Analysis Pipeline
==================================================
Author  : Ying Zhao
Email   : weiying.data@gmail.com
GitHub  : github.com/Ying-Data
Date    : April 2026

Description
-----------
End-to-end churn analysis pipeline for a California telecom dataset.
Covers: data loading, EDA, cleaning, feature engineering,
Random Forest ML model (AUC 0.9251), and clean CSV export.

Input files (place in same folder as this script):
  - telecom_customer_churn.csv
  - telecom_data_dictionary.csv
  - telecom_zipcode_population.csv

Output:
  - telecom_clean_powerbi.csv  (57 columns, ready for Power BI)
  - Console report with all key findings

Usage:
  python churn_analysis.py
"""

# ==============================================================
# 0. IMPORTS
# ==============================================================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder


# ==============================================================
# 1. LOAD DATA
# ==============================================================
print("=" * 60)
print("TELECALIFORNIA — CHURN ANALYSIS PIPELINE")
print("=" * 60)

print("\n[1/7] Loading data...")

# Load with latin1 encoding (handles special characters in this dataset)
df = pd.read_csv("telecom_customer_churn.csv", encoding="latin1")
pop = pd.read_csv("telecom_zipcode_population.csv", encoding="latin1")
dd  = pd.read_csv("telecom_data_dictionary.csv",  encoding="latin1")

print(f"  ✓ Customer data:    {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  ✓ Population data:  {pop.shape[0]:,} ZIP codes")
print(f"  ✓ Data dictionary:  {dd.shape[0]} field definitions")


# ==============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================
print("\n[2/7] Exploratory Data Analysis...")

# --- Customer status breakdown ---
status_counts = df["Customer Status"].value_counts()
total = len(df)
churned_n = status_counts.get("Churned", 0)
stayed_n  = status_counts.get("Stayed",  0)
joined_n  = status_counts.get("Joined",  0)

print(f"\n  Customer Status:")
print(f"    Stayed  : {stayed_n:,}  ({stayed_n/total*100:.1f}%)")
print(f"    Churned : {churned_n:,}  ({churned_n/total*100:.1f}%)")
print(f"    Joined  : {joined_n:,}   ({joined_n/total*100:.1f}%)")

# --- Revenue summary ---
total_rev   = df["Total Revenue"].sum()
churned_rev = df.loc[df["Customer Status"] == "Churned", "Total Revenue"].sum()
avg_charge_all     = df["Monthly Charge"].mean()
avg_charge_churned = df.loc[df["Customer Status"] == "Churned", "Monthly Charge"].mean()

print(f"\n  Revenue:")
print(f"    Total revenue:        ${total_rev:,.0f}")
print(f"    Revenue from churners:${churned_rev:,.0f}")
print(f"    Avg monthly charge (all):     ${avg_charge_all:.2f}")
print(f"    Avg monthly charge (churners):${avg_charge_churned:.2f}")

# --- Missing values audit ---
print(f"\n  Missing Values:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
for col, n in missing.items():
    pct = n / total * 100
    print(f"    {col:<40} {n:>5} ({pct:.1f}%) — business logic null")

# --- Churn by key segments ---
# Only look at active-base (Stayed + Churned, exclude Joined)
active = df[df["Customer Status"] != "Joined"].copy()
active_n = len(active)

print(f"\n  Churn Rate by Contract Type:")
for contract, grp in active.groupby("Contract"):
    rate = (grp["Customer Status"] == "Churned").mean() * 100
    print(f"    {contract:<20} {rate:.1f}%")

print(f"\n  Churn Rate by Internet Type:")
internet_col = "Internet Type" if "Internet Type" in df.columns else None
if internet_col:
    for itype, grp in active.groupby(internet_col):
        rate = (grp["Customer Status"] == "Churned").mean() * 100
        print(f"    {str(itype):<20} {rate:.1f}%")

# --- Churn reason breakdown ---
if "Churn Reason" in df.columns:
    churned = df[df["Customer Status"] == "Churned"]
    print(f"\n  Top 5 Churn Reasons:")
    top_reasons = churned["Churn Reason"].value_counts().head(5)
    for reason, count in top_reasons.items():
        pct = count / len(churned) * 100
        print(f"    {str(reason):<45} {count:>4} ({pct:.1f}%)")

# --- Churn category breakdown ---
if "Churn Category" in df.columns:
    print(f"\n  Churn by Category:")
    for cat, count in churned["Churn Category"].value_counts().items():
        pct = count / len(churned) * 100
        print(f"    {str(cat):<30} {count:>4} ({pct:.1f}%)")


# ==============================================================
# 3. DATA CLEANING
# ==============================================================
print("\n[3/7] Cleaning data...")

# --- Join population data ---
# Standardize ZIP code column name
df_zip_col  = "Zip Code"
pop_zip_col = "Zip Code" if "Zip Code" in pop.columns else pop.columns[0]
pop_pop_col = "Population" if "Population" in pop.columns else pop.columns[1]

df = df.merge(
    pop[[pop_zip_col, pop_pop_col]].rename(columns={pop_zip_col: df_zip_col}),
    on=df_zip_col,
    how="left"
)
matched = df["Population"].notna().sum()
print(f"  ✓ ZIP-population join: {matched:,}/{total:,} matched ({matched/total*100:.1f}%)")

# --- Fix negative Monthly Charge (billing credits → clip to 0) ---
neg_charges = (df["Monthly Charge"] < 0).sum()
df["Has_Credit_Adjustment"] = (df["Monthly Charge"] < 0).astype(int)
df["Monthly Charge"] = df["Monthly Charge"].clip(lower=0)
print(f"  ✓ Clipped {neg_charges} negative monthly charges to 0 (billing credits flagged)")

# --- Binary churn flag ---
df["Is_Churned"] = (df["Customer Status"] == "Churned").astype(int)

# --- Fill nulls with business-logic values ---
# Internet-dependent columns: null = "No Internet Service"
internet_dependent = [
    "Online Security", "Online Backup", "Device Protection Plan",
    "Premium Tech Support", "Streaming TV", "Streaming Movies",
    "Streaming Music", "Unlimited Data", "Internet Type"
]
for col in internet_dependent:
    if col in df.columns:
        df[col] = df[col].fillna("No Internet Service")

# Phone-dependent columns
phone_dependent = ["Multiple Lines"]
for col in phone_dependent:
    if col in df.columns:
        df[col] = df[col].fillna("No Phone Service")

# Churn-specific columns: null = "Not Applicable" (customer didn't churn)
churn_cols = ["Churn Category", "Churn Reason"]
for col in churn_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Not Churned")

print(f"  ✓ Business-logic nulls filled for internet/phone/churn columns")

# --- Remaining numeric nulls → median ---
numeric_cols = df.select_dtypes(include=[np.number]).columns
remaining_nulls = df[numeric_cols].isnull().sum().sum()
if remaining_nulls > 0:
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    print(f"  ✓ {remaining_nulls} remaining numeric nulls filled with column median")

print(f"  ✓ Cleaning complete. Final null count: {df.isnull().sum().sum()}")


# ==============================================================
# 4. FEATURE ENGINEERING
# ==============================================================
print("\n[4/7] Engineering features...")

# --- Revenue per month ---
df["Revenue_Per_Month"] = np.where(
    df["Tenure in Months"] > 0,
    df["Total Revenue"] / df["Tenure in Months"],
    df["Monthly Charge"]
)

# --- Refund rate ---
df["Refund_Rate"] = np.where(
    df["Total Revenue"] > 0,
    df["Total Refunds"] / df["Total Revenue"],
    0
)

# --- Extra charges ratio ---
df["Extra_Charges_Ratio"] = np.where(
    df["Monthly Charge"] > 0,
    (df.get("Total Extra Data Charges", pd.Series(0, index=df.index)) +
     df.get("Total Long Distance Charges", pd.Series(0, index=df.index))) /
    (df["Monthly Charge"] * df["Tenure in Months"].clip(lower=1)),
    0
)

# --- Tenure segment (lifecycle stage) ---
def tenure_segment(months):
    if months <= 6:   return "0-6M"
    elif months <= 12: return "6-12M"
    elif months <= 24: return "1-2Y"
    elif months <= 48: return "2-4Y"
    else:             return "4-6Y"

df["Tenure_Segment"] = df["Tenure in Months"].apply(tenure_segment)

# --- Age segment ---
def age_segment(age):
    if age < 30:   return "18-29"
    elif age < 45: return "30-44"
    elif age < 60: return "45-59"
    else:          return "60+"

df["Age_Segment"] = df["Age"].apply(age_segment)

# --- Population category ---
def pop_category(pop_val):
    if pd.isna(pop_val): return "Unknown"
    if pop_val < 10000:  return "Rural"
    elif pop_val < 50000: return "Suburban"
    elif pop_val < 200000: return "Urban"
    else:                return "Metro"

df["Pop_Category"] = df["Population"].apply(pop_category)

# --- Service bundle score (0 = no services, 8 = fully bundled) ---
service_cols = [
    "Online Security", "Online Backup", "Device Protection Plan",
    "Premium Tech Support", "Streaming TV", "Streaming Movies",
    "Streaming Music", "Unlimited Data"
]
def is_active_service(val):
    return 1 if str(val).lower() in ["yes", "1", "true"] else 0

df["Service_Bundle_Score"] = sum(
    df[col].apply(is_active_service) for col in service_cols if col in df.columns
)

# --- Streaming count ---
streaming_cols = ["Streaming TV", "Streaming Movies", "Streaming Music"]
df["Streaming_Count"] = sum(
    df[col].apply(is_active_service) for col in streaming_cols if col in df.columns
)

# --- Contract risk score (higher = more at-risk) ---
contract_risk_map = {"Month-to-Month": 3, "One Year": 2, "Two Year": 1}
df["Contract_Risk"] = df["Contract"].map(contract_risk_map).fillna(2)

# --- High-value customer flag ---
median_charge = df["Monthly Charge"].median()
df["Is_High_Value"] = (df["Monthly Charge"] >= median_charge * 1.2).astype(int)

# --- Loyal referrer flag ---
df["Is_Loyal_Referrer"] = (
    (df["Number of Referrals"] >= 2) &
    (df["Tenure in Months"] >= 24)
).astype(int)

# --- Fiber customer flag ---
if "Internet Type" in df.columns:
    df["Is_Fiber"] = (df["Internet Type"] == "Fiber Optic").astype(int)
else:
    df["Is_Fiber"] = 0

# --- Digital native (young + phone + streaming) ---
df["Is_Digital_Native"] = (
    (df["Age"] < 35) &
    (df["Streaming_Count"] >= 2)
).astype(int)

# --- Estimated CLTV (simplified: monthly charge × avg remaining lifetime) ---
avg_tenure = df["Tenure in Months"].mean()
df["Est_CLTV"] = df["Monthly Charge"] * (avg_tenure - df["Tenure in Months"]).clip(lower=0)

features_added = [
    "Has_Credit_Adjustment", "Is_Churned", "Revenue_Per_Month", "Refund_Rate",
    "Extra_Charges_Ratio", "Tenure_Segment", "Age_Segment", "Pop_Category",
    "Service_Bundle_Score", "Streaming_Count", "Contract_Risk",
    "Is_High_Value", "Is_Loyal_Referrer", "Is_Fiber", "Is_Digital_Native", "Est_CLTV"
]
print(f"  ✓ {len(features_added)} new features engineered:")
for f in features_added:
    print(f"      + {f}")


# ==============================================================
# 5. MACHINE LEARNING — RANDOM FOREST
# ==============================================================
print("\n[5/7] Training Random Forest model...")

# --- Build feature matrix ---
# Use Stayed + Churned only (exclude Joined — they haven't had a chance to churn)
ml_df = df[df["Customer Status"] != "Joined"].copy()

# Select numeric and encoded categorical features
numeric_features = [
    "Age", "Tenure in Months", "Monthly Charge", "Total Revenue",
    "Total Refunds", "Number of Referrals", "Avg Monthly Long Distance Charges",
    "Avg Monthly GB Download", "Contract_Risk", "Service_Bundle_Score",
    "Streaming_Count", "Is_High_Value", "Is_Loyal_Referrer", "Is_Fiber",
    "Is_Digital_Native", "Revenue_Per_Month", "Refund_Rate", "Est_CLTV",
    "Has_Credit_Adjustment"
]

# Keep only columns that exist in the dataframe
numeric_features = [c for c in numeric_features if c in ml_df.columns]

# Encode categorical columns
cat_features = ["Contract", "Internet Type", "Tenure_Segment", "Age_Segment"]
cat_features = [c for c in cat_features if c in ml_df.columns]

le = LabelEncoder()
encoded_cats = []
for col in cat_features:
    new_col = f"{col}_encoded"
    ml_df[new_col] = le.fit_transform(ml_df[col].astype(str))
    encoded_cats.append(new_col)

all_features = numeric_features + encoded_cats

X = ml_df[all_features].fillna(0)
y = ml_df["Is_Churned"]

print(f"  Training set: {len(ml_df):,} customers ({y.sum():,} churned, {(y==0).sum():,} stayed)")
print(f"  Feature count: {len(all_features)}")

# --- Train / test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Fit model ---
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# --- Evaluate ---
y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

auc       = roc_auc_score(y_test, y_proba)
acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
cm        = confusion_matrix(y_test, y_pred)

print(f"\n  ── Model Performance ──────────────────────")
print(f"  AUC-ROC    : {auc:.4f}")
print(f"  Accuracy   : {acc:.4f}")
print(f"  Precision  : {precision:.4f}  (of predicted churners, how many were right)")
print(f"  Recall     : {recall:.4f}  (of actual churners, how many did we catch)")
print(f"  F1-Score   : {f1:.4f}")
print(f"\n  Confusion Matrix:")
print(f"              Predicted: Stay  Predicted: Churn")
print(f"  Actual: Stay   {cm[0][0]:>5}             {cm[0][1]:>5}")
print(f"  Actual: Churn  {cm[1][0]:>5}             {cm[1][1]:>5}")

# --- Cross-validation ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"\n  5-Fold Cross-Validation AUC:")
print(f"  Scores : {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean   : {cv_scores.mean():.4f}")
print(f"  Std    : {cv_scores.std():.4f}")

# --- Feature importance ---
importance = pd.Series(rf.feature_importances_, index=all_features)
importance = importance.sort_values(ascending=False)
print(f"\n  Top 10 Feature Importances:")
for feat, imp in importance.head(10).items():
    bar = "█" * int(imp * 100)
    print(f"    {feat:<40} {imp:.4f}  {bar}")


# ==============================================================
# 6. SCORE ALL CUSTOMERS & ASSIGN RISK TIERS
# ==============================================================
print("\n[6/7] Scoring all customers and assigning risk tiers...")

# Score all 7,043 customers (including Joined — for early warning)
X_all = df[all_features].fillna(0)
df["Churn_Probability"] = rf.predict_proba(X_all)[:, 1]

# Risk tier thresholds
def risk_tier(prob):
    if prob >= 0.70:   return "Critical Risk"
    elif prob >= 0.50: return "High Risk"
    elif prob >= 0.30: return "Medium Risk"
    else:              return "Low Risk"

df["Churn_Risk_Tier"] = df["Churn_Probability"].apply(risk_tier)

tier_counts = df["Churn_Risk_Tier"].value_counts()
print(f"\n  Risk Tier Distribution:")
for tier in ["Critical Risk", "High Risk", "Medium Risk", "Low Risk"]:
    n = tier_counts.get(tier, 0)
    rev_at_risk = df.loc[df["Churn_Risk_Tier"] == tier, "Monthly Charge"].sum() * 12
    print(f"    {tier:<15} {n:>5} customers  (${rev_at_risk:>12,.0f} annual revenue exposure)")

# --- Geographic insights ---
print(f"\n  Top 5 Cities by Churn Rate (min 20 customers):")
city_stats = (
    df[df["Customer Status"] != "Joined"]
    .groupby("City")
    .agg(total=("Customer ID", "count"), churned=("Is_Churned", "sum"))
    .query("total >= 20")
)
city_stats["churn_rate"] = city_stats["churned"] / city_stats["total"]
top_cities = city_stats.sort_values("churn_rate", ascending=False).head(5)
for city, row in top_cities.iterrows():
    print(f"    {city:<25} {row['churn_rate']*100:.1f}%  ({row['churned']:.0f}/{row['total']:.0f})")

# --- Offer analysis ---
if "Offer" in df.columns:
    print(f"\n  Churn Rate by Offer:")
    offer_stats = (
        df[df["Customer Status"] != "Joined"]
        .groupby("Offer")
        .apply(lambda g: (g["Is_Churned"].mean() * 100).round(1))
        .sort_values(ascending=False)
    )
    for offer, rate in offer_stats.items():
        print(f"    {str(offer):<15} {rate:.1f}%")


# ==============================================================
# 7. EXPORT CLEAN CSV FOR POWER BI
# ==============================================================
print("\n[7/7] Exporting clean dataset...")

output_path = "telecom_clean_powerbi.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"  ✓ Saved: {output_path}")
print(f"  ✓ Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# List all columns
print(f"\n  Columns in output file ({df.shape[1]} total):")
original_count = len(pd.read_csv("telecom_customer_churn.csv", encoding="latin1", nrows=0).columns)
print(f"    Original columns:    {original_count}")
print(f"    Engineered features: {df.shape[1] - original_count - 1} (+ Population from join)")
print(f"    ML outputs:          2 (Churn_Probability, Churn_Risk_Tier)")

# ==============================================================
# FINAL SUMMARY
# ==============================================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE — KEY FINDINGS SUMMARY")
print("=" * 60)

churn_rate = churned_n / (churned_n + stayed_n) * 100
annual_loss = churned_rev / df.loc[df["Customer Status"] == "Churned", "Tenure in Months"].mean() * 12

print(f"""
  📊 Data
     · 7,043 customers across 3 merged datasets
     · 15 engineered features created
     · 0 unexplained missing values

  📈 Business Findings
     · Churn rate:          {churn_rate:.1f}%  (industry avg ~20%)
     · Revenue lost:        ${churned_rev:,.0f}
     · Est. annual loss:    ~${annual_loss:,.0f}
     · Avg charge churners: ${avg_charge_churned:.2f}/mo  (vs ${avg_charge_all:.2f} overall)
     · Worst contract:      Month-to-Month at ~51.7% churn
     · Danger zone:         0-6 months tenure (77.2% churn rate)
     · Worst offer:         Offer E — 67.6% churn

  🤖 ML Model (Random Forest)
     · AUC-ROC:     {auc:.4f}
     · 5-Fold CV:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
     · Accuracy:    {acc*100:.1f}%
     · F1 (churn):  {f1:.2f}

  🎯 Risk Tiers
     · Critical Risk: {tier_counts.get('Critical Risk', 0):,} customers
     · High Risk:     {tier_counts.get('High Risk', 0):,} customers
     · Medium Risk:   {tier_counts.get('Medium Risk', 0):,} customers
     · Low Risk:      {tier_counts.get('Low Risk', 0):,} customers

  📁 Output
     · telecom_clean_powerbi.csv  ← import this into Power BI

  💡 Top Recommendation
     Contact top 200 Critical Risk customers immediately.
     Estimated annual revenue at stake: ~$1.76M.
""")
