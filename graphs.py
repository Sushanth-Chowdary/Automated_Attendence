import os
import pandas as pd
import matplotlib.pyplot as plt

results_dir = 'Results'

# --- 1. Load the Excel Data ---
excel_path = os.path.join(results_dir, 'Final_Report_With_Efficiency.xlsx')
df_excel = pd.read_excel(excel_path, header=1)

student_col = df_excel.columns[0]
overall_col = df_excel.columns[-1]

df_excel = df_excel.dropna(subset=[student_col])
df_excel = df_excel[df_excel[student_col] != 'Student Name & Roll No']
df_excel = df_excel[df_excel[student_col] != 'AVERAGE EFFICIENCY']

def parse_percentage(val):
    if pd.isna(val) or val == 'Efficiency': return 0.0
    if isinstance(val, str):
        return float(val.replace('%', '').strip())
    return float(val)

df_excel['Full_Name'] = df_excel[student_col].astype(str).str.strip()
df_excel['Accuracy'] = df_excel[overall_col].apply(parse_percentage)

df_excel['Roll_Name'] = df_excel['Full_Name'].apply(lambda x: x.split(' - ')[0].strip())


# --- GRAPH 1: Accuracy vs Person Line Graph ---
plt.figure(figsize=(14, 7))
plt.plot(df_excel['Full_Name'], df_excel['Accuracy'], marker='o', linestyle='-', color='b')
plt.title('Overall Accuracy by Person')
plt.xlabel('Person')
plt.ylabel('Overall Accuracy (%)')
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()

graph1_path = os.path.join(results_dir, 'accuracy_vs_person.png')
plt.savefig(graph1_path)
print(f"Saved Graph 1: {graph1_path}")
plt.close()


# --- GRAPH 2: Accuracy vs Label Size Dual-Axis Graph ---
try:
    csv_path = os.path.join(results_dir, 'dataset_comparison.csv')
    mapping_path = os.path.join(results_dir, 'name_mapping.csv')
    
    df_counts = pd.read_csv(csv_path)
    df_map = pd.read_csv(mapping_path)
    
    df_merged = pd.merge(df_excel, df_map, on='Roll_Name', how='inner')
    
    df_merged = pd.merge(df_merged, df_counts, left_on='Short_Name', right_on='Name', how='inner')
    
    df_merged = df_merged.sort_values('Original Count (LABELS)', ascending=False)
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Primary Y-Axis (Bar Chart for Sample Sizes)
    ax1.bar(df_merged['Full_Name'], df_merged['Original Count (LABELS)'], color='lightblue', label='Sample Size')
    ax1.set_xlabel('Person')
    ax1.set_ylabel('Sample Size (Count)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax1.set_xticklabels(df_merged['Full_Name'], rotation=90)
    
    # Secondary Y-Axis (Line Chart for Accuracies)
    ax2 = ax1.twinx()
    ax2.plot(df_merged['Full_Name'], df_merged['Accuracy'], color='red', marker='o', linewidth=2, label='Accuracy (%)')
    ax2.set_ylabel('Accuracy (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 105)
    
    plt.title('Accuracy vs Person Label Size')
    fig.tight_layout()
    
    graph2_path = os.path.join(results_dir, 'accuracy_vs_label_size.png')
    plt.savefig(graph2_path)
    print(f"Saved Graph 2: {graph2_path}")
    plt.close()
    
except FileNotFoundError as e:
    print(f"\nError: Could not find required files. Details: {e}")
    print("Please ensure both 'dataset_comparison.csv' and 'name_mapping.csv' are located in the Results/ directory.")