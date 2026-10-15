import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the absolute path to the Results folder where data is stored and graphs will be saved
results_dir = '/mnt/DATA/SUSHANTH/EE23B044/Results'

# --- 1. Load the Excel Data ---
excel_path = os.path.join(results_dir, 'Final_Report_With_Efficiency.xlsx')
# Read the excel file, using row 1 as headers
df_excel = pd.read_excel(excel_path, header=1)

# Drop any empty rows or the header sub-row
df_excel = df_excel.dropna(subset=[df_excel.columns[0]])
df_excel = df_excel[df_excel[df_excel.columns[0]] != 'Student Name & Roll No']

student_col = df_excel.columns[0]
overall_col = df_excel.columns[-1]

names = df_excel[student_col].astype(str).str.strip()

# Helper to convert "93.06%" into a clean float 93.06
def parse_percentage(val):
    if pd.isna(val) or val == 'Efficiency': return 0.0
    if isinstance(val, str):
        return float(val.replace('%', '').strip())
    return float(val)

accuracies = df_excel[overall_col].apply(parse_percentage)

# --- GRAPH 1: Accuracy vs Person Line Graph ---
plt.figure(figsize=(14, 7))
plt.plot(names, accuracies, marker='o', linestyle='-', color='b')
plt.title('Overall Accuracy by Person')
plt.xlabel('Person')
plt.ylabel('Overall Accuracy (%)')
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()

# Save Graph 1 to the Results folder
graph1_path = os.path.join(results_dir, 'accuracy_vs_person.png')
plt.savefig(graph1_path)
print(f"Saved Graph 1: {graph1_path}")
plt.show()

# --- GRAPH 2: Accuracy vs Label Size Dual-Axis Graph ---
try:
    # Load the CSV generated from count_samples.py located in Results folder
    csv_path = os.path.join(results_dir, 'dataset_comparison.csv')
    df_counts = pd.read_csv(csv_path)
    
    # Create a clean DataFrame for the accuracy data to merge
    df_acc = pd.DataFrame({'Name': names, 'Accuracy': accuracies})
    
    # Merge the counts and accuracies on the 'Name' column
    df_merged = pd.merge(df_counts, df_acc, on='Name', how='inner')
    
    # Sort by sample size descending for a cleaner visual layout
    df_merged = df_merged.sort_values('Original Count (LABELS)', ascending=False)
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Plot Sample Size as a Bar Chart on the primary Y-axis
    ax1.bar(df_merged['Name'], df_merged['Original Count (LABELS)'], color='lightblue', label='Sample Size')
    ax1.set_xlabel('Person')
    ax1.set_ylabel('Sample Size (Count)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Ignore warning for strict tick label mapping 
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax1.set_xticklabels(df_merged['Name'], rotation=90)
    
    # Plot Accuracy as a Line Chart on a secondary Y-axis
    ax2 = ax1.twinx()
    ax2.plot(df_merged['Name'], df_merged['Accuracy'], color='red', marker='o', linewidth=2, label='Accuracy (%)')
    ax2.set_ylabel('Accuracy (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 105) # Cap percentages at 100 with a slight buffer
    
    plt.title('Accuracy vs Person Label Size')
    fig.tight_layout()
    
    # Save Graph 2 to the Results folder
    graph2_path = os.path.join(results_dir, 'accuracy_vs_label_size.png')
    plt.savefig(graph2_path)
    print(f"Saved Graph 2: {graph2_path}")
    plt.show()
    
except FileNotFoundError:
    print(f"\nError: Could not find '{csv_path}'. Please ensure it is in the Results folder!")