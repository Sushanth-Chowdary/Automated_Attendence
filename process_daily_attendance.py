import os
import glob
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def process_daily_attendance(base_dir, target_folder):
    input_dir = os.path.join(base_dir, target_folder)
    
    # Define file paths
    true_attendance_path = os.path.join(base_dir, "filtered_true_attendance.csv")
    name_mapping_path = os.path.join(base_dir, "name_mapping.csv")
    
    # 1. Check if required files exist
    for path in [true_attendance_path, name_mapping_path]:
        if not os.path.exists(path):
            print(f"Error: Could not find required file at {path}")
            return

    # 2. Load True Attendance and Create Roll + Name Combo
    true_df = pd.read_csv(true_attendance_path)
    roll_col = true_df.iloc[:, 0].astype(str).str.strip()
    name_col = true_df.iloc[:, 1].astype(str).str.strip()
    
    true_df['Combo_Name'] = roll_col + " - " + name_col
    roll_to_combo = dict(zip(roll_col, true_df['Combo_Name']))
    true_df = true_df.set_index(roll_col)
    
    # Clean headers and cells
    true_df.columns = true_df.columns.astype(str).str.strip().str.replace('-', '')
    for col in true_df.columns:
        if col != 'Combo_Name':
            true_df[col] = true_df[col].astype(str).str.strip()
            
    # Load mappings
    name_map_df = pd.read_csv(name_mapping_path).dropna(subset=['Short_Name', 'Roll_Name'])
    name_map_df = name_map_df[name_map_df['Short_Name'].astype(str).str.strip() != '']
    short_to_roll = dict(zip(
        name_map_df['Short_Name'].astype(str).str.strip(),
        name_map_df['Roll_Name'].astype(str).str.strip()
    ))
    
    # 3. Scan for video output CSVs
    file_pattern = os.path.join(input_dir, "*_output.csv")
    output_files = glob.glob(file_pattern)
    
    if not output_files:
        print(f"No output CSV files found in {input_dir}")
        return

    # 4. Load all files and append a 'Date' column based on filename
    all_cam_data = []
    for f in output_files:
        filename = os.path.basename(f)
        date_str = filename.split('_')[0]
        
        df = pd.read_csv(f)
        df['Date'] = date_str
        all_cam_data.append(df)
        
    merged_cams = pd.concat(all_cam_data, ignore_index=True)
    
    # Clean names and map to Roll Numbers
    merged_cams['Name'] = merged_cams['Name'].astype(str).str.strip()
    merged_cams['Roll_Name'] = merged_cams['Name'].map(short_to_roll).fillna(merged_cams['Name'])
    
    # Ensure Detection Count is numeric
    merged_cams['Detection Count'] = pd.to_numeric(merged_cams['Detection Count'], errors='coerce').fillna(0)

    # 5. Group by Date and Roll_Name to get total daily detection count
    daily_counts = merged_cams.groupby(['Date', 'Roll_Name'])['Detection Count'].sum().reset_index()
    
    # Construct evaluation dataset matching Estimated with True Status
    eval_records = []
    unique_dates = sorted(daily_counts['Date'].unique())
    all_students = sorted(list(roll_to_combo.keys()))
    
    for date_str in unique_dates:
        if date_str not in true_df.columns:
            continue # Skip if no ground truth for this date
            
        for roll in all_students:
            true_status_raw = true_df.at[roll, date_str].lower() if roll in true_df.index else ""
            if true_status_raw in ['nan', 'unknown', 'none', '']:
                continue # Skip invalid true attendance
                
            true_is_present = (true_status_raw == 'present')
            
            # Get count (0 if student not detected at all that day)
            student_match = daily_counts[(daily_counts['Date'] == date_str) & (daily_counts['Roll_Name'] == roll)]
            count = student_match['Detection Count'].values[0] if not student_match.empty else 0
            
            eval_records.append({
                'Date': date_str,
                'Roll_Name': roll,
                'Total_Count': count,
                'True_Present': true_is_present,
                'True_Label': 'Present' if true_is_present else 'Absent'
            })
            
    eval_df = pd.DataFrame(eval_records)
    
    if eval_df.empty:
        print("Error: No valid matching ground truth data found to evaluate efficiency.")
        return

    # 6. Dynamically Find Best Threshold (Testing 1 through 20)
    best_threshold = 1
    best_accuracy = 0.0
    
    for t in range(1, 21):
        eval_df['Est_Present'] = eval_df['Total_Count'] >= t
        accuracy = (eval_df['Est_Present'] == eval_df['True_Present']).mean() * 100
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t
            
    print(f"Optimal Threshold Found: {best_threshold} total detections/day (Overall Accuracy: {best_accuracy:.2f}%)")

    # 7. Apply Best Threshold and Build Final Pivot Table
    eval_df['Estimated'] = eval_df['Total_Count'].apply(lambda x: 'Present' if x >= best_threshold else 'Absent')
    
    idx = pd.Index(all_students, name='Roll_Name')
    combined_df = pd.DataFrame(index=idx)
    bottom_summary_row = {}
    
    total_matches_overall = 0
    total_possible_overall = 0

    for date_str in unique_dates:
        date_data = eval_df[eval_df['Date'] == date_str].set_index('Roll_Name')
        
        count_series = date_data['Total_Count']
        est_series = date_data['Estimated']
        true_series = date_data['True_Label']
        
        # Calculate date efficiency
        matches = (date_data['Estimated'] == date_data['True_Label']).sum()
        total = len(date_data)
        
        if total > 0:
            date_eff = (matches / total) * 100
            total_matches_overall += matches
            total_possible_overall += total
            bottom_summary_row[(date_str, 'Estimated')] = f"{date_eff:.2f}%"
        else:
            bottom_summary_row[(date_str, 'Estimated')] = "N/A"
            
        bottom_summary_row[(date_str, 'Total Count')] = ""
        bottom_summary_row[(date_str, 'True')] = ""
        
        combined_df[(date_str, 'Total Count')] = count_series.reindex(idx).fillna(0).astype(int)
        combined_df[(date_str, 'Estimated')] = est_series.reindex(idx).fillna("")
        combined_df[(date_str, 'True')] = true_series.reindex(idx).fillna("")

    # --- ADD STUDENT-WISE ACCURACY ---
    student_matches = pd.Series(0.0, index=combined_df.index)
    student_valid = pd.Series(0.0, index=combined_df.index)

    for date_str in unique_dates:
        est_col = combined_df[(date_str, 'Estimated')]
        true_col = combined_df[(date_str, 'True')]
        
        valid_mask = (true_col != "")
        matches = (est_col == true_col) & valid_mask
        
        student_matches += matches.astype(int)
        student_valid += valid_mask.astype(int)

    student_acc = pd.Series(pd.NA, index=combined_df.index)
    mask = student_valid > 0
    student_acc[mask] = (student_matches[mask] / student_valid[mask]) * 100
    
    combined_df[('Overall Summary', 'Student Accuracy')] = student_acc.apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
    )

    overall_eff_val = f"{(total_matches_overall / total_possible_overall) * 100:.2f}%" if total_possible_overall > 0 else "N/A"
    bottom_summary_row[('Overall Summary', 'Student Accuracy')] = overall_eff_val

    # Structure Columns and Index
    combined_df.columns = pd.MultiIndex.from_tuples(combined_df.columns)
    combined_df.index = combined_df.index.map(lambda x: roll_to_combo.get(str(x), str(x)))
    combined_df.index.name = 'Student Name & Roll No'
    combined_df = combined_df.sort_index()
    
    # Append Summary Row
    summary_df = pd.DataFrame([bottom_summary_row], index=['AVERAGE EFFICIENCY'])
    summary_df.columns = combined_df.columns 
    combined_df = pd.concat([combined_df, summary_df])

    # 8. Save Files
    csv_output = os.path.join(input_dir, "Final_Daily_Report_With_Efficiency.csv")
    xlsx_output = os.path.join(input_dir, "Final_Daily_Report_With_Efficiency.xlsx")

    print(f"Saving merged CSV and formatting Excel report using threshold >= {best_threshold}...")
    combined_df.to_csv(csv_output)
    combined_df.to_excel(xlsx_output)

    # 9. Excel Formatting
    wb = openpyxl.load_workbook(xlsx_output)
    ws = wb.active
    ws.title = "Daily Attendance Efficiency"

    ws['A1'] = 'Student Name & Roll No'
    ws.merge_cells('A1:A2')
    ws.delete_rows(3)  
    
    max_row = ws.max_row
    max_col = ws.max_column

    header_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    green_font = Font(color="006100")
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    red_font = Font(color="9C0006")
    summary_fill = PatternFill(start_color="D5F5E3", end_color="D5F5E3", fill_type="solid")
    center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left_align = Alignment(horizontal="left", vertical="center")
    thin_border = Border(left=Side(style='thin', color='BDC3C7'), right=Side(style='thin', color='BDC3C7'), 
                         top=Side(style='thin', color='BDC3C7'), bottom=Side(style='thin', color='BDC3C7'))

    # Format Headers
    for row in range(1, 3):
        for cell in ws[row]:
            try:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = center_align
                cell.border = thin_border
            except AttributeError:
                pass
                
    # Merge the Overall Summary Header properly
    ws.merge_cells(start_row=1, start_column=max_col, end_row=1, end_column=max_col)

    # Format Data Rows
    for row_idx in range(3, max_row):
        for col_idx, cell in enumerate(ws[row_idx], start=1):
            try:
                cell.alignment = left_align if col_idx == 1 else center_align
                cell.border = thin_border
                
                if row_idx % 2 == 0:
                    cell.fill = PatternFill(start_color="F8F9F9", end_color="F8F9F9", fill_type="solid")
                    
                val = str(cell.value).strip().lower()
                if val == 'present':
                    cell.fill = green_fill
                    cell.font = green_font
                elif val == 'absent':
                    cell.fill = red_fill
                    cell.font = red_font
                elif val.endswith('%') and val != 'n/a':
                    if float(val.strip('%')) < 80.0:
                        cell.fill = red_fill
                        cell.font = red_font
            except (AttributeError, ValueError):
                pass

    # Format the Summary Row at the bottom
    for col_idx, cell in enumerate(ws[max_row], start=1):
        try:
            cell.alignment = center_align if col_idx > 1 else left_align
            cell.border = thin_border
            cell.fill = summary_fill
            cell.font = Font(bold=True)
            
            val = str(cell.value).strip().lower()
            if val.endswith('%') and val != 'n/a':
                if float(val.strip('%')) < 80.0:
                    cell.fill = red_fill
                    cell.font = Font(color="9C0006", bold=True)
        except (AttributeError, ValueError):
            pass

    # Merge Estimated, Count, True columns for the summary row percentages
    for i in range(len(unique_dates)):
        start_col = 2 + (i * 3)
        end_col = start_col + 2
        ws.merge_cells(start_row=max_row, start_column=start_col, end_row=max_row, end_column=end_col)
        
    # Append the Overall global efficiency below the date efficiencies
    overall_row = max_row + 1
    ws.cell(row=overall_row, column=1, value="OVERALL TOTAL EFFICIENCY")
    ws.cell(row=overall_row, column=2, value=overall_eff_val)
    ws.merge_cells(start_row=overall_row, start_column=2, end_row=overall_row, end_column=max_col)
    
    for cell in ws[overall_row]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True, size=12)
        cell.alignment = center_align
        cell.border = thin_border
    ws.cell(row=overall_row, column=1).alignment = left_align

    # Adjust column widths
    for col_idx in range(1, max_col + 1):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = 15 if col_idx < max_col else 20
    ws.column_dimensions['A'].width = 38

    ws.freeze_panes = "B3"
    wb.save(xlsx_output)
    
    print(f"\nSuccess! Reports generated:")
    print(f" -> {csv_output}")
    print(f" -> {xlsx_output}")

if __name__ == "__main__":
    BASE_DIRECTORY = "ATTENDENCE RESULTS/Results"
    TARGET_FOLDER = "test 7"  
    
    process_daily_attendance(BASE_DIRECTORY, TARGET_FOLDER)