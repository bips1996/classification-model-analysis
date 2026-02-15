import os
import pandas as pd
import urllib.request
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split

def download_bank_marketing_dataset():
    """
    Downloads the Bank Marketing dataset from UCI ML Repository
    Dataset: Bank Marketing (with bank client data)
    Source: UCI Machine Learning Repository
    """
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Dataset URL from UCI ML Repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    
    print("📥 Downloading Bank Marketing dataset from UCI ML Repository...")
    
    try:
        # Download the zip file
        zip_path = data_dir / 'bank-additional.zip'
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Read the full dataset (bank-additional-full.csv has more data)
        data_file = data_dir / 'bank-additional' / 'bank-additional-full.csv'
        df = pd.read_csv(data_file, sep=';')
        
        # Convert target variable 'y' to binary (0=no, 1=yes)
        df['target'] = (df['y'] == 'yes').astype(int)
        df = df.drop('y', axis=1)
        
        # Encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes
        
        # Save as CSV
        output_path = data_dir / 'bank_marketing.csv'
        df.to_csv(output_path, index=False)
        
        # Create train-test split
        print("\n📂 Creating train-test split...")
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save split data
        X_train.to_csv(data_dir / 'X_train.csv', index=False)
        X_test.to_csv(data_dir / 'X_test.csv', index=False)
        y_train.to_csv(data_dir / 'y_train.csv', index=False)
        y_test.to_csv(data_dir / 'y_test.csv', index=False)
        
        print(f"✅ Train-test split completed!")
        print(f"  - Training: {X_train.shape[0]} samples")
        print(f"  - Testing: {X_test.shape[0]} samples")
        
        print(f"\n✅ Dataset processed successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"💾 Saved to: {output_path}")
        print(f"\nDataset info:")
        print(f"  - Features: {df.shape[1] - 1}")
        print(f"  - Instances: {df.shape[0]}")
        print(f"  - Classification: Binary (client subscribed term deposit)")
        print(f"  - Target distribution:")
        print(df['target'].value_counts())
        
        # Clean up temporary files
        if zip_path.exists():
            os.remove(zip_path)
        import shutil
        if (data_dir / 'bank-additional').exists():
            shutil.rmtree(data_dir / 'bank-additional')
        
        return True
        
    except Exception as e:
        print(f" Error downloading dataset: {e}")
        print("\n Alternative: Download manually from:")
        print("https://archive.ics.uci.edu/ml/datasets/Bank+Marketing")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("📦 Bank Marketing Dataset Downloader")
    print("=" * 60)
    
    # Check if dataset and split files already exist
    split_files_exist = all([
        Path('data/X_train.csv').exists(),
        Path('data/X_test.csv').exists(),
        Path('data/y_train.csv').exists(),
        Path('data/y_test.csv').exists()
    ])
    
    if Path('data/bank_marketing.csv').exists() and split_files_exist:
        print("Dataset and split files already exist!")
        df = pd.read_csv('data/bank_marketing.csv')
        print(f"📊 Dataset shape: {df.shape}")
        X_train = pd.read_csv('data/X_train.csv')
        print(f"✅ Training samples: {X_train.shape[0]}")
    else:
        # Try to download Bank Marketing dataset
        success = download_bank_marketing_dataset()
        
        if not success:
            print("\n" + "=" * 60)
            print("📖 Manual Download Instructions:")
            print("=" * 60)
            print("Visit: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing")
            print("Download and place 'bank-additional-full.csv' in data/ folder")
    
    print("\n✨ Ready to proceed with model training!")
