import pandas as pd

# Load dataset
df = pd.read_csv('jobs.csv')

# Columns used
columns_to_use = ['Key Skills', 'Job Title', 'Role Category', 'Functional Area', 'Industry']

# Fill missing values
for col in columns_to_use:
    df[col] = df[col].fillna('').astype(str)

# Remove duplicate job titles
df = df.drop_duplicates(subset=['Job Title'])

# Convert to lowercase
for col in columns_to_use:
    df[col] = df[col].str.lower()

# Create combined text column
df['tags'] = (
    df['Job Title'] + " " +
    df['Key Skills'] + " " +
    df['Role Category'] + " " +
    df['Industry']
)

# Save processed dataset
df.to_csv('processed_jobs.csv', index=False)

print("Dataset preprocessing completed successfully.")