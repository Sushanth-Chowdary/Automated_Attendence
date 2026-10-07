import os
import glob
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def process_attendance(base_dir, target_folder):
    input_dir = os.path.join(base_dir, target_folder)
    
    # Define file paths
    true_attendance_path = os.path.join(base_dir, "filtered_true_attendance.csv")
    name_mapping_path = os.path.join(base_dir, "name_mapping.csv")
    protocol_mapping_path = os.path.join(base_dir, "protocol_mapping.csv")
    
    # 1. Check if required files exist
    for path in [true_attendance_path, name_mapping_path, protocol_mapping_path]:
        if not os.path.exists(path):
            print(f"Error: Could not find required file at {path}")
            return

    # 2. Load True Attendance and Mappings safely as string types
    true_df = pd.read_csv(true_attendance_path)
    true_df.iloc[:, 0] = true_df.iloc[:, 0].astype(str).str.strip()
    true_df = true_df.set_index(true_df.columns[0])
    # Ensure column headers (dates) are string type for reliable key matching
    true_df.columns = true_df.columns.astype(str).str.strip()
    
    name_map_df = pd.read_csv(name_mapping_path).dropna(subset=['Short_Name', 'Roll_Name'])
    # Filter out empty or whitespace-only rows
    name_map_df = name_map_df[name_map_df['Short_Name'].astype(str).str.strip() != '']
    short_to_roll = dict(zip(
        name_map_df['Short_Name'].astype(str).str.strip(),
        name_map_df['Roll_Name'].astype(str).str.strip()
    ))
    
    protocol_map_df = pd.read_csv(protocol_mapping_path)
    base_to_protocol = dict(zip(
        protocol_map_df['Base_Timestamp'].astype(str).str.strip(),
        protocol_map_df['Protocol_Length'].astype(str).str.strip()
    ))
    
    # 3. Scan for video output CSVs
    file_pattern = os.path.join(input_dir, "*_output.csv")
    output_files = glob.glob(file_pattern)
    
    if not output_files:
        print(f"No output CSV files found in {input_dir}")
        return

    # 4. Group files by base timestamp (to pair cam1 and cam2)
    sessions = {}
    for file in output_files:
        filename = os.path.basename(file)
        # Assuming filename format: 20260828_110647_cam1_raw_output.csv
        parts = filename.split('_')
        if len(parts) >= 2:
            base_timestamp = f"{parts[0]}_{parts[1]}"
            if base_timestamp not in sessions:
                sessions[base_timestamp] = []
            sessions[base_timestamp].append(file)

    dfs = []
    timestamps_ordered = sorted(list(sessions.keys()))

    print("Processing grouped camera files and calculating union...")
    
    for ts in timestamps_ordered:
        date_str = ts.split('_')[0]
        protocol = base_to_protocol.get(ts, "Unknown Length")
        
        session_files = sessions[ts]
        
        # Read and merge cam1 and cam2 (or just one if the other is missing)
        cam_dfs = [pd.read_csv(f) for f in session_files]
        merged_cams = pd.concat(cam_dfs, ignore_index=True)
        
        # Safely map Short Names to Roll Names
        merged_cams['Name'] = merged_cams['Name'].astype(str).str.strip()
        merged_cams['Roll_Name'] = merged_cams['Name'].map(short_to_roll).fillna(merged_cams['Name'])
        
        # Determine logical Union (Present if ANY camera says Present)
        merged_cams['Is_Present'] = merged_cams['Status'].astype(str).str.strip().str.lower() == 'present'
        
        grouped = merged_cams.groupby('Roll_Name').agg(
            Is_Present=('Is_Present', 'any'),
            Detection_Count=('Detection Count', 'sum')  # Sum counts from both cams
        ).reset_index()
        
        grouped['Estimated'] = grouped['Is_Present'].apply(lambda x: 'Present' if x else 'Absent')
        grouped = grouped.rename(columns={'Detection_Count': 'Count'})
        grouped = grouped.set_index('Roll_Name')
        
        # Fetch True Attendance for the specific date
        if date_str in true_df.columns:
            grouped['True'] = true_df[date_str]
        else:
            grouped['True'] = "Unknown" 
            
        grouped = grouped[['Estimated', 'Count', 'True']]
        
        # Setup 3-tier MultiIndex: Date -> Timestamp (Protocol) -> Data Columns
        header_level_2 = f"{ts} ({protocol})"
        grouped.columns = pd.MultiIndex.from_product([[date_str], [header_level_2], grouped.columns])
        dfs.append(grouped)

    # 5. Combine everything into one large DataFrame
    combined_df = pd.concat(dfs, axis=1)
    combined_df.index.name = 'Student Name & Roll No'
    combined_df = combined_df.sort_index()
    
    # 6. Calculate Efficiency across all protocols
    student_matches = pd.Series(0, index=combined_df.index)
    valid_sessions = len(timestamps_ordered)
    total_students = len(combined_df)
    
    date_efficiencies = {}
    
    for ts in timestamps_ordered:
        date_str = ts.split('_')[0]
        protocol = base_to_protocol.get(ts, "Unknown Length")
        col_group = f"{ts} ({protocol})"
        
        est_col = combined_df[(date_str, col_group, 'Estimated')].astype(str).str.strip().str.lower()
        true_col = combined_df[(date_str, col_group, 'True')].astype(str).str.strip().str.lower()
        
        # Calculate matching accuracy
        matches = (est_col == true_col) & (~true_col.isin(['unknown', 'nan', 'none', '']))
        student_matches += matches.astype(int)
        
        valid_student_count = (~true_col.isin(['unknown', 'nan', 'none', ''])).sum()
        date_accuracy = (matches.sum() / valid_student_count * 100) if valid_student_count > 0 else 0.0
        
        date_efficiencies[(date_str, col_group, 'Estimated')] = f"{date_accuracy:.2f}%"
        date_efficiencies[(date_str, col_group, 'Count')] = ""
        date_efficiencies[(date_str, col_group, 'True')] = ""

    # Add overall efficiency column
    combined_df[('Overall', 'Summary', 'Efficiency')] = (student_matches / valid_sessions * 100).apply(lambda x: f"{x:.2f}%")
    
    # Add bottom summary row
    eff_df = pd.DataFrame([date_efficiencies], index=['EFFICIENCY'])
    total_matches_all = student_matches.sum()
    total_possible_data_points = total_students * valid_sessions
    eff_df[('Overall', 'Summary', 'Efficiency')] = f"{(total_matches_all / total_possible_data_points * 100):.2f}%" 
    
    combined_df = pd.concat([combined_df, eff_df])

    # 7. Save Files
    csv_output = os.path.join(input_dir, "Final_Report_With_Efficiency.csv")
    xlsx_output = os.path.join(input_dir, "Final_Report_With_Efficiency.xlsx")

    print("Saving merged CSV and formatting Excel report...")
    combined_df.to_csv(csv_output)
    combined_df.to_excel(xlsx_output)

    # 8. Excel Formatting for 3-Tier MultiIndex
    wb = openpyxl.load_workbook(xlsx_output)
    ws = wb.active
    ws.title = "Attendance Efficiency"

    # Cleanup the Pandas 3-tier index export quirks
    ws['A1'] = 'Student Name & Roll No'
    ws.merge_cells('A1:A3')
    ws.delete_rows(4)  # Removes the blank row generated beneath the headers
    
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

    # Format Headers (Rows 1, 2, and 3)
    for row in range(1, 4):
        for cell in ws[row]:
            try:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = center_align
                cell.border = thin_border
            except AttributeError:
                pass

    # Format Data Rows
    for row_idx in range(4, max_row):
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
                elif val.endswith('%'):
                    if float(val.strip('%')) < 80.0:
                        cell.fill = red_fill
                        cell.font = red_font
            except (AttributeError, ValueError):
                pass

    # Format the bottom DATE EFFICIENCY row
    for col_idx, cell in enumerate(ws[max_row], start=1):
        try:
            cell.alignment = center_align
            cell.border = thin_border
            cell.fill = summary_fill
            cell.font = Font(bold=True)
            
            val = str(cell.value).strip().lower()
            if val.endswith('%') and float(val.strip('%')) < 80.0:
                cell.fill = red_fill
                cell.font = Font(color="9C0006", bold=True)
        except (AttributeError, ValueError):
            pass

    # Dynamic Merging for the bottom row (Merging Estimated, Count, True)
    for i in range(len(timestamps_ordered)):
        start_col = 2 + (i * 3)
        end_col = start_col + 2
        ws.merge_cells(start_row=max_row, start_column=start_col, end_row=max_row, end_column=end_col)

    # Adjust column widths
    for col_idx in range(1, max_col + 1):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = 15
    ws.column_dimensions['A'].width = 35

    ws.freeze_panes = "B4"
    wb.save(xlsx_output)
    
    print(f"\nSuccess! Reports generated:")
    print(f" -> {csv_output}")
    print(f" -> {xlsx_output}")

if __name__ == "__main__":
    BASE_DIRECTORY = "ATTENDENCE RESULTS/Results"
    TARGET_FOLDER = "test 1"  # Update with your target test folder name
    
    process_attendance(BASE_DIRECTORY, TARGET_FOLDER)