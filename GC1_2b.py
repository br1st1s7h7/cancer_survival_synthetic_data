import pandas as pd
import numpy as np

from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer


folder = r"C:\Users\brist\OneDrive\Documents\Cancer_data\\"
input_file = (folder + "cancer_patient_records_main.csv")
output_file = (folder + "synthetic_2b_gaussian_copula_1M.csv")


df = pd.read_csv(input_file)


drop_cols = [
    "Record_ID",
    "PatientID",
    "Location_ID",
    "County_Year_Key",
    "State_Year_Key",
    "Survey_Batch_ID",
    "Data_Entry_Clerk_ID",
    "Notes",
    "Diagnosis_Date",
    "Chemotherapy_Cycles"
]

df = df.drop(columns=drop_cols)


df["Sex"] = (
    df["Sex"]
    .astype(str)
    .str.strip()
    .str.upper()
    .replace({
        "MALE": "Male",
        "FEMALE": "Female"
    })
)


numeric_cols = [
    "Year",
    "Age",
    "Tumor_Size_mm",
    "BMI",
    
    "Distance_to_Treatment_Center_km",
    "Follow_up_Months"
]


categorical_cols = [
    "State",
    "County",
    "Age_Group",
    "Sex",
    "Ethnicity",
    "Cancer_Type",
    "Stage_at_Diagnosis",
    "Metastasis_Status",
    "Smoking_Status",
    "Alcohol_Consumption",
    "Diabetes_Status",
    "Family_History_Cancer",
    "Genetic_Marker_Status",
    "Insurance_Type",
    "Treatment_Type",
    "Hospital_Type",
    "Record_Source",
    "Survival_Status"
]


for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())


for col in categorical_cols:
    df[col] = df[col].astype("object")

    if df[col].isna().any():
        if col in [
            "Smoking_Status",
            "Alcohol_Consumption",
            "Insurance_Type"
        ]:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna("Unknown")


clinical_limits = {
    "Age": (18, 96),
    "BMI": (15, 50),
    "Tumor_Size_mm": (0, 150),
    "Distance_to_Treatment_Center_km": (0, 200),
    "Follow_up_Months": (0, 120),
    "Chemotherapy_Cycles": (0, 12)
}


for col, (lower, upper) in clinical_limits.items():
    df[col] = df[col].clip(lower=lower, upper=upper)


df["Year"] = df["Year"].astype(str)


metadata = Metadata.detect_from_dataframe(
    data=df
)

metadata.update_column(
    column_name="State",
    sdtype="categorical"
)

metadata.update_column(
    column_name="Year",
    sdtype="categorical"
)


synthesizer = GaussianCopulaSynthesizer(
    metadata,
    enforce_min_max_values=True,
    enforce_rounding=True
)


synthesizer.fit(df)


synthetic = synthesizer.sample(
    num_rows=1_000_000
)


synthetic["Year"] = synthetic["Year"].astype(int)

synthetic["Age"] = synthetic["Age"].round().astype(int)

synthetic["Chemotherapy_Cycles"] = (
    synthetic["Chemotherapy_Cycles"]
    .round()
    .astype(int)
)


for col, (lower, upper) in clinical_limits.items():
    synthetic[col] = synthetic[col].clip(
        lower=lower,
        upper=upper
    )


synthetic.to_csv(
    output_file,
    index=False
)


print("Real data shape:", df.shape)
print("Synthetic data shape:", synthetic.shape)
print("Saved to:", output_file)