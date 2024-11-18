from langchain_community.document_loaders.csv_loader import CSVLoader

import os
import pandas as pd  # Added for handling CSV data

# Initialize the tool with your specific CSV file
csv_path = 'healthcare_dataset.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"The specified CSV file does not exist: {csv_path}")

# Load only the first 50 rows
df = pd.read_csv(csv_path).head(5)
print('by pandas-->',df)
# df.to_csv('healthcare_dataset.csv', index=False)


# def read_and_load_csv(file_path, limit=5):
#     loader = CSVLoader(file_path=file_path)
#     documents = loader.load()
#     print('by loader-->',documents)
#     return documents[:limit]
# read_and_load_csv(csv_path,limit=5)

def read_and_load_csv(file_path, limit=5):
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()[:limit]  # Slice before printing
    print('by loader-->', documents)  # Print only the limited documents
    return documents

read_and_load_csv(csv_path, limit=5)
