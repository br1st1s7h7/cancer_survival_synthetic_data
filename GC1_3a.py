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

print("Real data shape:", df.shape)
print("Missing values:", df.isna().sum().sum())


metadata_3 = Metadata.detect_from_dataframe(
    data=df
)


metadata_3.update_column(
    column_name="State",
    sdtype="categorical"
)


metadata_3.update_column(
    column_name="Year",
    sdtype="categorical"
)


metadata_3.validate()


print("\nMetadata validation successful.")


synthesizer_3_gc = GaussianCopulaSynthesizer(
    metadata_3,
    enforce_min_max_values=True,
    enforce_rounding=True
)

synthesizer_3_gc.fit(df)

synthetic_3_gc = synthesizer_3_gc.sample(
    num_rows=1_000_000
)


print("\nSynthetic data shape:", synthetic_3_gc.shape)


print(synthetic_3_gc.head())


print("\nSynthetic data types:")
print(synthetic_3_gc.dtypes)


print("\nMissing values:")
print(synthetic_3_gc.isna().sum().sum())


output_file = folder + "synthetic_3_gaussian_copula_1M.csv"

synthetic_3_gc.to_csv(
    output_file,
    index=False
)


print("\nSaved:", output_file)