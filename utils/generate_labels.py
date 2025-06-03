import re
import os
import pandas as pd

uaspeech_severity_map = {
    "F02": "Severe",
    "F03": "Very_Severe",
    "F04": "Moderate",
    "F05": "Mild",
    "M08": "Mild",
    "M09": "Mild",
    "M10": "Mild",
    "M14": "Mild",
    "M11": "Moderate",
    "M16": "Severe",
    "M04": "Very_Severe",
    "M12": "Very_Severe",
    "M05": "Moderate",
    "M07": "Severe",
    "M01": "Very_Severe",
    "CF02": "Healthy",
    "CF03": "Healthy",
    "CF04": "Healthy",
    "CF05": "Healthy",
    "CM01": "Healthy",
    "CM04": "Healthy",
    "CM05": "Healthy",
    "CM06": "Healthy",
    "CM08": "Healthy",
    "CM09": "Healthy",
    "CM10": "Healthy",
    "CM12": "Healthy",
    "CM13": "Healthy"
} 

torgo_severity_map = {
    "FC01": "Healthy", 
    "FC02": "Healthy",  
    "FC03": "Healthy",
    "MC01": "Healthy",  
    "MC02": "Healthy",  
    "MC03": "Healthy",
    "MC04": "Healthy",
    "F01": "Moderate", 
    "F03": "Mild",  
    "F04": "Mild",
    "M01": "Severe", 
    "M02": "Severe", 
    "M03": "Mild", 
    "M04": "Severe", 
    "M05": "Moderate"
}
    
def extract_metadata(dataset_path, dataset_name):
    """
    Extract metadata from audio files in the dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        dataset_name: Name of the dataset ('uaspeech' or 'torgo')
        
    Returns:
        List of dictionaries containing metadata for each audio file
    """
    metadata = []
    print(f"[DEBUG] Searching for {dataset_name} files in {dataset_path}")
    
    for root, _, files in os.walk(dataset_path):
        if dataset_name.lower() in root.lower():  # Case-insensitive path check
            for f in files:
                if f.endswith('.wav'):
                    wav_path = os.path.join(root, f)
                    speaker = os.path.basename(os.path.dirname(wav_path))
                    severity = "Unknown"
                    
                    if dataset_name.lower() == "uaspeech":
                        speaker = speaker.strip().upper()
                        severity = uaspeech_severity_map.get(speaker, "Unknown")
                        print(f"[DEBUG] UASpeech: Found speaker {speaker} with severity {severity}")
                    
                    elif dataset_name.lower() == "torgo":
                        match = re.search(r'(?:wav_(?:arrayMic|headMic)_)?((?:FC|MC|F|M)\d{2})(?:S\d+)?', f)
                        if match:
                            speaker = next(g for g in match.groups() if g).upper()
                        else:
                            speaker = os.path.basename(os.path.dirname(wav_path)).upper()
                        severity = torgo_severity_map.get(speaker, "Unknown")
                        print(f"[DEBUG] TORGO: Found speaker {speaker} with severity {severity}")
                    
                    if severity != "Unknown":
                        metadata.append({
                            "file_path": wav_path,
                            "speaker": speaker,
                            "dataset": dataset_name,
                            "severity": severity
                        })
    
    print(f"[DEBUG] Found {len(metadata)} valid files for {dataset_name}")
    return metadata

def main(data_dir=None):
    """
    Generate metadata CSV file for the datasets.
    
    Args:
        data_dir: Path to the directory containing both datasets. If None, uses environment variable or default.
    """
    if data_dir is None:
        data_dir = os.getenv('DYSARTHRIA_DATA_DIR', "audio")
    
    print(f"[DEBUG] Using data directory: {data_dir}")
    
    # Extract metadata from both datasets
    torgo_data = extract_metadata(data_dir, "torgo")
    uaspeech_data = extract_metadata(data_dir, "uaspeech")
    
    # Combine metadata
    combined = torgo_data + uaspeech_data
    df = pd.DataFrame(combined)
    
    # Filter by severity
    severity_classes = ['Mild', 'Moderate', 'Severe', 'Very_Severe']
    df = df[df['severity'].isin(severity_classes)]
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total files: {len(df)}")
    print("\nSeverity distribution:")
    print(df['severity'].value_counts())
    print("\nDataset distribution:")
    print(df['dataset'].value_counts())
    print("\nSpeaker distribution:")
    print(df['speaker'].value_counts())
    
    # Save metadata
    output_file = "verified_metadata.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file}")
    print(f"Found {len(df)} valid files")

if __name__ == "__main__":
    main("C:\\Users\\Sowmya\\research\\dysarthria_pipeline3\\data")