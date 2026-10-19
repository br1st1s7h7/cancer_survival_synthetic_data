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
    df[col] = df[col].fillna(
        df[col].median()
    )


categorical_cols = [
    "Smoking_Status",
    "Alcohol_Consumption",
    "Insurance_Type"
]


for col in categorical_cols:
    df[col] = df[col].fillna(
        df[col].mode()[0]
    )





print("Real data shape:", df.shape)
print("Missing values:", df.isna().sum().sum())


metadata_2a = Metadata.detect_from_dataframe(
    data=df
)


metadata_2a.update_column(
    column_name="State",
    sdtype="categorical"
)


metadata_2a.update_column(
    column_name="Year",
    sdtype="categorical"
)


metadata_2a.validate()


print("\nMetadata validation successful.")


synthesizer_2a_gc = GaussianCopulaSynthesizer(
    metadata_2a,
    enforce_min_max_values=True,
    enforce_rounding=True
)

synthesizer_2a_gc.fit(df)

synthetic_2a_gc = synthesizer_2a_gc.sample(
    num_rows=1_000_000
)


print("\nSynthetic data shape:", synthetic_2a_gc.shape)


print(synthetic_2a_gc.head())


print("\nSynthetic data types:")
print(synthetic_2a_gc.dtypes)


print("\nMissing values:")
print(synthetic_2a_gc.isna().sum().sum())


output_file = folder + "synthetic_2a_gaussian_copula_1M.csv"

synthetic_2a_gc.to_csv(
    output_file,
    index=False
)


print("\nSaved:", output_file)