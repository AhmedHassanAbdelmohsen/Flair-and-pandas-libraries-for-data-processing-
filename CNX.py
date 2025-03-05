import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence

# Load the Excel file
file_path = r"C:\Users\Ahmed Hassan\Desktop\conc_proj\CNX.xlsx"
xls = pd.ExcelFile(file_path)

# Read tables
df1 = xls.parse("Customer Survey Results")
df2 = xls.parse("Account Interactions")

# Merge on Interaction ID and remove duplicate columns
merged_df = df1.merge(df2, on="Interaction ID", how="inner")

# Rename the column
merged_df.rename(columns={"verbatim": "Verbatims"}, inplace=True)

# Remove duplicate rows
merged_df.drop_duplicates(inplace=True)

# Load Flair sentiment classifier
classifier = TextClassifier.load("sentiment")

# Sentiment analysis function
def get_flair_sentiment(text):
    """Returns Positive, Negative, or Neutral using Flair."""
    if pd.isna(text):  # Handle NaN values
        return "neutral"
    sentence = Sentence(text)
    classifier.predict(sentence)
    return sentence.labels[0].value.lower()  # Returns 'positive', 'negative', or 'neutral'

# Apply sentiment analysis
merged_df["Sentiment"] = merged_df["Verbatims"].astype(str).apply(get_flair_sentiment)

# Reorder columns to place Sentiment after Verbatims
column_order = ["Interaction ID", "Verbatims", "Sentiment"] + [
    col for col in merged_df.columns if col not in ["Interaction ID", "Verbatims", "Sentiment"]
]
merged_df = merged_df[column_order]

# Save the file as CSV
output_path = r"C:\Users\Ahmed Hassan\Desktop\conc_proj\CNX_T.csv"
merged_df.to_csv(output_path, index=False, encoding="utf-8")

print(f"File saved successfully at: {output_path}")
