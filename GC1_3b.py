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


df = df.dropna().drop_duplicates()



winsor_cols = [
    "Tumor_Size_mm",
    "BMI",
    "Distance_to_Treatment_Center_km",
    "Follow_up_Months"
]


for col in winsor_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)

    df[col] = df[col].clip(
        lower=lower,
        upper=upper
    )


print("Real data shape:", df.shape)
print("Missing values:", df.isna().sum().sum())


metadata_3b = Metadata.detect_from_dataframe(
    data=df
)


metadata_3b.update_column(
    column_name="State",
    sdtype="categorical"
)


metadata_3b.update_column(
    column_name="Year",
    sdtype="categorical"
)


metadata_3b.validate()


print("\nMetadata validation successful.")


synthesizer_3b_gc = GaussianCopulaSynthesizer(
    metadata_3b,
    enforce_min_max_values=True,
    enforce_rounding=True
)

synthesizer_3b_gc.fit(df)

synthetic_3b_gc = synthesizer_3b_gc.sample(
    num_rows=1_000_000
)


print("\nSynthetic data shape:", synthetic_3b_gc.shape)


print(synthetic_3b_gc.head())


print("\nSynthetic data types:")
print(synthetic_3b_gc.dtypes)


print("\nMissing values:")
print(synthetic_3b_gc.isna().sum().sum())


output_file = folder + "synthetic_3b_gaussian_copula_1M.csv"

synthetic_3b_gc.to_csv(
    output_file,
    index=False
)


print("\nSaved:", output_file)