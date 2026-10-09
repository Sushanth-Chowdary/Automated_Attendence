import os
import pandas as pd

def generate_error_report(base_dir, target_folder):
    input_dir = os.path.join(base_dir, target_folder)
    excel_filename = os.path.join(input_dir, "Final_Report_With_Efficiency.xlsx")
    output_txt = os.path.join(input_dir, "Error_Report.txt")
    
    print(f"Loading data from {excel_filename}...")
    
    if not os.path.exists(excel_filename):
        print(f"Error: Could not find {excel_filename}")
        return
        
    # Load the Excel file with the 3-tier header
    df = pd.read_excel(excel_filename, header=[0, 1, 2], index_col=0)
    
    # Filter out the summary rows at the bottom
    df = df[~df.index.astype(str).str.contains('AVERAGE|EFFICIENCY|Summary', case=False, na=False)]
    
    false_negatives = {}
    false_positives = {}
    
    # Extract all unique (Date, Session) combinations from the columns
    date_session_pairs = []
    for col in df.columns:
        if col[2] == 'Estimated': 
            date_session_pairs.append((col[0], col[1]))
            
    # Analyze each video/session
    for date, session in date_session_pairs:
        estimated_col = (date, session, 'Estimated')
        true_col = (date, session, 'True')
        
        if estimated_col not in df.columns or true_col not in df.columns:
            continue
            
        # Standardize text for accurate comparison
        est_data = df[estimated_col].astype(str).str.strip().str.lower()
        true_data = df[true_col].astype(str).str.strip().str.lower()
        
        # 1. False Negatives: True is 'present', but Estimated is 'absent'
        fn_mask = (true_data == 'present') & (est_data == 'absent')
        fn_students = df.index[fn_mask].tolist()
        if fn_students:
            false_negatives[session] = fn_students
            
        # 2. False Positives: True is 'absent', but Estimated is 'present'
        fp_mask = (true_data == 'absent') & (est_data == 'present')
        fp_students = df.index[fp_mask].tolist()
        if fp_students:
            false_positives[session] = fp_students

    # --- FORMAT AND OUTPUT THE RESULTS ---
    output_lines = []
    
    output_lines.append("========== FALSE NEGATIVES ==========")
    output_lines.append("(Student was ACTUALLY PRESENT, but AI marked ABSENT)\n")
    if not false_negatives:
        output_lines.append("No False Negatives found.\n")
    else:
        for session, students in false_negatives.items():
            output_lines.append(f"Video/Session: {session}")
            for student in students:
                output_lines.append(f"  - {student}")
            output_lines.append("")

    output_lines.append("\n========== FALSE POSITIVES ==========")
    output_lines.append("(Student was ACTUALLY ABSENT, but AI marked PRESENT)\n")
    if not false_positives:
        output_lines.append("No False Positives found.\n")
    else:
        for session, students in false_positives.items():
            output_lines.append(f"Video/Session: {session}")
            for student in students:
                output_lines.append(f"  - {student}")
            output_lines.append("")

    # Print to terminal
    full_report = "\n".join(output_lines)
    print(full_report)
    
    # Save to text file in the target folder
    with open(output_txt, "w") as f:
        f.write(full_report)
    print(f"\n[+] Report successfully saved to '{output_txt}'")

if __name__ == "__main__":
    BASE_DIRECTORY = "ATTENDENCE RESULTS/Results"
    TARGET_FOLDER = "test 1" 
    
    generate_error_report(BASE_DIRECTORY, TARGET_FOLDER)