import pandas as pd
import sdv

from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer


folder = r"C:\Users\brist\OneDrive\Documents\Cancer_data\\"

df = pd.read_csv(
    folder + "cancer_patient_records_main.csv"
)


df = df.drop(
    columns=[
        "Record_ID",
        "PatientID",
        "Location_ID",
        "County_Year_Key",
        "State_Year_Key",
        "Survey_Batch_ID",
        "Data_Entry_Clerk_ID",
        "Notes",
        "Diagnosis_Date"
    ],
    errors="ignore"
)


df["Sex"] = df["Sex"].replace({
    "MALE": "Male",
    "FEMALE": "Female"
})


numeric_cols = [
    "Tumor_Size_mm",
    "BMI",
    "Distance_to_Treatment_Center_km",
    "Follow_up_Months"
]


for col in numeric_cols:
    df[col + "_Missing"] = df[col].isna().astype(int)


categorical_cols = [
    "Smoking_Status",
    "Alcohol_Consumption",
    "Insurance_Type"
]


for col in categorical_cols:
    df[col] = df[col].fillna("Missing")


df["Tumor_Size_mm"] = df["Tumor_Size_mm"].fillna(
    df["Tumor_Size_mm"].median()
)

df["BMI"] = df["BMI"].fillna(
    df["BMI"].median()
)

df["Distance_to_Treatment_Center_km"] = df[
    "Distance_to_Treatment_Center_km"
].fillna(
    df["Distance_to_Treatment_Center_km"].median()
)

df["Follow_up_Months"] = df[
    "Follow_up_Months"
].fillna(
    df["Follow_up_Months"].median()
)


df["Age"] = df["Age"].clip(
    lower=18,
    upper=96
)

df["BMI"] = df["BMI"].clip(
    lower=15,
    upper=50
)

df["Tumor_Size_mm"] = df["Tumor_Size_mm"].clip(
    lower=0,
    upper=150
)

df["Distance_to_Treatment_Center_km"] = df[
    "Distance_to_Treatment_Center_km"
].clip(
    lower=0,
    upper=200
)

df["Follow_up_Months"] = df[
    "Follow_up_Months"
].clip(
    lower=0,
    upper=120
)

df["Chemotherapy_Cycles"] = df[
    "Chemotherapy_Cycles"
].clip(
    lower=0,
    upper=12
)


df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[17, 24, 34, 44, 54, 64, 74, 200],
    labels=[
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65-74",
        "75+"
    ]
).astype(str)


df["Year"] = df["Year"].astype(str)


print("Real data shape:", df.shape)
print("Missing values:", df.isna().sum().sum())


metadata_1b = Metadata.detect_from_dataframe(
    data=df
)


metadata_1b.update_column(
    column_name="State",
    sdtype="categorical"
)


metadata_1b.update_column(
    column_name="Year",
    sdtype="categorical"
)


metadata_1b.validate()


print("\nMetadata validation successful.")


synthesizer_1b_gc = GaussianCopulaSynthesizer(
    metadata_1b,
    enforce_min_max_values=True,
    enforce_rounding=True
)


print("\nTraining Gaussian Copula...")

synthesizer_1b_gc.fit(df)

print("Gaussian Copula training completed.")


print("\nGenerating 1,000,000 synthetic records...")

synthetic_1b_gc = synthesizer_1b_gc.sample(
    num_rows=1_000_000
)


missing_indicator_cols = [
    col + "_Missing"
    for col in numeric_cols
]

synthetic_1b_gc = synthetic_1b_gc.drop(
    columns=missing_indicator_cols,
    errors="ignore"
)


synthetic_1b_gc["Year"] = synthetic_1b_gc["Year"].astype(int)


synthetic_1b_gc["Age_Group"] = pd.cut(
    synthetic_1b_gc["Age"],
    bins=[17, 24, 34, 44, 54, 64, 74, 200],
    labels=[
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65-74",
        "75+"
    ]
).astype(str)


print("\nSynthetic data shape:", synthetic_1b_gc.shape)

print("\nFirst 5 rows:")
print(synthetic_1b_gc.head())


print("\nSynthetic data types:")
print(synthetic_1b_gc.dtypes)


print("\nMissing values:")
print(synthetic_1b_gc.isna().sum().sum())


output_file = folder + "synthetic_1b_gaussian_copula_1M.csv"

synthetic_1b_gc.to_csv(
    output_file,
    index=False
)


print("\nSaved:", output_file)
