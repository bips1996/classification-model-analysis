import os
import pandas as pd
import urllib.request
import zipfile
from pathlib import Path

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
    
    # Check if dataset already exists
    if Path('data/bank_marketing.csv').exists():
        print("Dataset already exists at data/bank_marketing.csv")
        df = pd.read_csv('data/bank_marketing.csv')
        print(f"📊 Dataset shape: {df.shape}")
    else:
        # Try to download Bank Marketing dataset
        success = download_bank_marketing_dataset()
        
        if not success:
            print("\n" + "=" * 60)
            print("📖 Manual Download Instructions:")
            print("=" * 60)
            print("Visit: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing")
            print("Download and place 'bank-additional-full.csv' in data/ folder")
    
    print("\n✨ Ready to proceed with data exploration!")
