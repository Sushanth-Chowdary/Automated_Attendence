import os
import glob
import re
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def load_reference_data(base_dir):
    """Loads filtered_true_attendance.csv, name_mapping.csv, protocol_mapping.csv, and dataset_comparison.csv."""
    true_attendance_path = os.path.join(base_dir, "filtered_true_attendance.csv")
    name_mapping_path = os.path.join(base_dir, "name_mapping.csv")
    protocol_mapping_path = os.path.join(base_dir, "protocol_mapping.csv")
    dataset_comp_path = os.path.join(base_dir, "dataset_comparison.csv")

    for path in [true_attendance_path, name_mapping_path, protocol_mapping_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: Could not find required file at {path}")

    # 1. Load True Attendance
    true_df = pd.read_csv(true_attendance_path)
    roll_col = true_df.iloc[:, 0].astype(str).str.strip()
    name_col = true_df.iloc[:, 1].astype(str).str.strip()

    true_df['Combo_Name'] = roll_col + " - " + name_col
    roll_to_combo = dict(zip(roll_col, true_df['Combo_Name']))
    true_df = true_df.set_index(roll_col)
    
    # Prevent crashes if there are duplicate roll numbers in true_df
    true_df = true_df[~true_df.index.duplicated(keep='first')]

    true_df.columns = true_df.columns.astype(str).str.strip().str.replace('-', '')
    for col in true_df.columns:
        if col != 'Combo_Name':
            true_df[col] = true_df[col].astype(str).str.strip()

    # 2. Load Name Mapping
    name_map_df = pd.read_csv(name_mapping_path).dropna(subset=['Short_Name', 'Roll_Name'])
    name_map_df = name_map_df[name_map_df['Short_Name'].astype(str).str.strip() != '']
    short_to_roll = dict(zip(
        name_map_df['Short_Name'].astype(str).str.strip(),
        name_map_df['Roll_Name'].astype(str).str.strip()
    ))
    short_to_roll_lower = {k.lower(): v for k, v in short_to_roll.items()}

    # 3. Load Protocol Mapping
    protocol_map_df = pd.read_csv(protocol_mapping_path)
    base_to_protocol = dict(zip(
        protocol_map_df['Base_Timestamp'].astype(str).str.strip(),
        protocol_map_df['Protocol_Length'].astype(str).str.strip()
    ))

    # 4. Load Dataset Comparison (Label Sizes)
    roll_to_labels = {}
    if os.path.exists(dataset_comp_path):
        comp_df = pd.read_csv(dataset_comp_path)
        name_col_comp = comp_df.columns[0]
        count_col_comp = comp_df.columns[1]
        for _, row in comp_df.iterrows():
            s_name = str(row[name_col_comp]).strip()
            cnt = pd.to_numeric(row[count_col_comp], errors='coerce')
            cnt = int(cnt) if pd.notna(cnt) else 0
            r_name = short_to_roll.get(s_name, short_to_roll_lower.get(s_name.lower(), s_name))
            roll_to_labels[r_name] = roll_to_labels.get(r_name, 0) + cnt
    else:
        print(f"Warning: {dataset_comp_path} not found. Label Size will be 'N/A'.")

    # Safety: Guarantee ANY mapped roll number exists in roll_to_combo even if missing from true_df
    for roll in short_to_roll.values():
        if roll not in roll_to_combo:
            roll_to_combo[roll] = str(roll)
    for roll in roll_to_labels.keys():
        if roll not in roll_to_combo:
            roll_to_combo[roll] = str(roll)

    return true_df, roll_to_combo, short_to_roll, base_to_protocol, roll_to_labels


def is_cam1_file(filepath):
    """Returns True if the CSV file belongs to cam1."""
    fname = os.path.basename(filepath).lower()
    return 'cam1' in fname or 'cam_1' in fname


# ==============================================================================
# A1 & A3: PROTOCOL-WISE ATTENDANCE
# ==============================================================================
def run_protocol_attendance(output_files, true_df, roll_to_combo, short_to_roll, base_to_protocol, roll_to_labels, cam1_only=False):
    selected_files = [f for f in output_files if is_cam1_file(f)] if cam1_only else list(output_files)

    sessions = {}
    for file in selected_files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 2:
            base_timestamp = f"{parts[0]}_{parts[1]}"
            sessions.setdefault(base_timestamp, []).append(file)

    timestamps_ordered = sorted(list(sessions.keys()))
    if not timestamps_ordered:
        return pd.DataFrame(), {}

    dfs = []
    for ts in timestamps_ordered:
        date_str = ts.split('_')[0]
        protocol = base_to_protocol.get(ts, "Unknown Length")
        col_group = f"{ts} ({protocol})"

        session_files = sessions[ts]
        cam_dfs = [pd.read_csv(f) for f in session_files]
        merged_cams = pd.concat(cam_dfs, ignore_index=True)

        merged_cams['Name'] = merged_cams['Name'].astype(str).str.strip()
        merged_cams['Roll_Name'] = merged_cams['Name'].map(short_to_roll).fillna(merged_cams['Name'])
        merged_cams['Detection Count'] = pd.to_numeric(merged_cams['Detection Count'], errors='coerce').fillna(0)
        merged_cams['Is_Present'] = merged_cams['Status'].astype(str).str.strip().str.lower() == 'present'

        grouped = merged_cams.groupby('Roll_Name').agg(
            Is_Present=('Is_Present', 'any'),
            Detection_Count=('Detection Count', 'sum')
        ).reset_index()

        grouped['Estimated'] = grouped['Is_Present'].apply(lambda x: 'Present' if x else 'Absent')
        grouped = grouped.rename(columns={'Detection_Count': 'Count'})
        grouped = grouped.set_index('Roll_Name')

        if date_str in true_df.columns:
            grouped['True'] = true_df[date_str]
        else:
            grouped['True'] = ""

        grouped = grouped[['Estimated', 'Count', 'True']]
        grouped.columns = pd.MultiIndex.from_product([[date_str], [col_group], grouped.columns])
        dfs.append(grouped)

    combined_df = pd.concat(dfs, axis=1)
    
    # Extract unique indices and ensure all known students are represented
    all_students = sorted(list(roll_to_combo.keys()))
    idx = pd.Index(all_students, name='Roll_Name')
    combined_df = combined_df.reindex(idx)

    label_col = ('Dataset', 'Comparison', 'Label Size')
    label_series = pd.Series([roll_to_labels.get(str(r), "N/A") for r in idx], index=idx)
    label_df = pd.DataFrame({label_col: label_series}, index=idx)
    combined_df = pd.concat([label_df, combined_df], axis=1)

    overall_student_matches = pd.Series(0.0, index=idx)
    overall_student_valid = pd.Series(0.0, index=idx)
    protocol_student_matches = {}
    protocol_student_valid = {}
    protocol_stats = {}

    total_matches_overall = 0
    total_possible_overall = 0
    bottom_summary_row = {}

    valid_lbls = [v for v in label_series if isinstance(v, (int, float))]
    bottom_summary_row[label_col] = f"Avg: {sum(valid_lbls)/len(valid_lbls):.0f}" if valid_lbls else "N/A"

    for ts in timestamps_ordered:
        date_str = ts.split('_')[0]
        protocol = base_to_protocol.get(ts, "Unknown Length")
        col_group = f"{ts} ({protocol})"

        if protocol not in protocol_stats:
            protocol_stats[protocol] = {'matches': 0, 'total': 0}
            protocol_student_matches[protocol] = pd.Series(0.0, index=idx)
            protocol_student_valid[protocol] = pd.Series(0.0, index=idx)

        # Handle NaNs from the reindex safely
        est_col = combined_df[(date_str, col_group, 'Estimated')].fillna("Absent").astype(str).str.strip().str.lower()
        true_col = combined_df[(date_str, col_group, 'True')].fillna("").astype(str).str.strip().str.lower()

        valid_mask = (~true_col.isin(['unknown', 'nan', 'none', ''])) & (true_col != '')
        matches = (est_col == true_col) & valid_mask

        overall_student_matches += matches.astype(int)
        overall_student_valid += valid_mask.astype(int)
        protocol_student_matches[protocol] += matches.astype(int)
        protocol_student_valid[protocol] += valid_mask.astype(int)

        current_matches = matches.sum()
        current_total = valid_mask.sum()

        if current_total > 0:
            date_accuracy = (current_matches / current_total) * 100
            bottom_summary_row[(date_str, col_group, 'Estimated')] = f"{date_accuracy:.2f}%"
            protocol_stats[protocol]['matches'] += current_matches
            protocol_stats[protocol]['total'] += current_total
            total_matches_overall += current_matches
            total_possible_overall += current_total
        else:
            bottom_summary_row[(date_str, col_group, 'Estimated')] = "N/A"

        bottom_summary_row[(date_str, col_group, 'Count')] = ""
        bottom_summary_row[(date_str, col_group, 'True')] = ""

    def sort_key(p):
        m = re.search(r'\d+', str(p))
        return int(m.group()) if m else 999

    sorted_protocols = sorted(protocol_stats.keys(), key=sort_key)
    protocol_cols = []

    for prot in sorted_protocols:
        col_name = (f'{prot} Protocol', 'Summary', 'Efficiency')
        protocol_cols.append((prot, col_name))
        mask = protocol_student_valid[prot] > 0

        prot_series = pd.Series(pd.NA, index=idx)
        prot_series[mask] = (protocol_student_matches[prot][mask] / protocol_student_valid[prot][mask]) * 100

        combined_df[col_name] = prot_series.apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
        )

        p_matches = protocol_stats[prot]['matches']
        p_total = protocol_stats[prot]['total']
        bottom_summary_row[col_name] = f"{(p_matches / p_total) * 100:.2f}%" if p_total > 0 else "N/A"

    overall_col = ('Overall', 'Summary', 'Efficiency')
    mask_overall = overall_student_valid > 0
    overall_series = pd.Series(pd.NA, index=idx)
    overall_series[mask_overall] = (overall_student_matches[mask_overall] / overall_student_valid[mask_overall]) * 100

    combined_df[overall_col] = overall_series.apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
    )

    bottom_summary_row[overall_col] = f"{(total_matches_overall / total_possible_overall) * 100:.2f}%" if total_possible_overall > 0 else "N/A"

    combined_df.index = combined_df.index.map(lambda x: roll_to_combo.get(str(x), str(x)))
    combined_df.index.name = 'Student Name & Roll No'
    combined_df = combined_df.sort_index()

    summary_df = pd.DataFrame([bottom_summary_row], index=['AVERAGE EFFICIENCY'])
    summary_df.columns = combined_df.columns
    combined_df = pd.concat([combined_df, summary_df])

    info = {
        'num_sessions': len(timestamps_ordered),
        'protocol_cols': protocol_cols,
        'overall_col': overall_col
    }
    return combined_df, info


# ==============================================================================
# A2 & A4: DAILY COMBINED ATTENDANCE
# ==============================================================================
def run_daily_attendance(output_files, true_df, roll_to_combo, short_to_roll, roll_to_labels, cam1_only=False):
    selected_files = [f for f in output_files if is_cam1_file(f)] if cam1_only else list(output_files)
    if not selected_files:
        return pd.DataFrame(), {}

    all_cam_data = []
    for f in selected_files:
        filename = os.path.basename(f)
        date_str = filename.split('_')[0]
        df = pd.read_csv(f)
        df['Date'] = date_str
        all_cam_data.append(df)

    merged_cams = pd.concat(all_cam_data, ignore_index=True)
    merged_cams['Name'] = merged_cams['Name'].astype(str).str.strip()
    merged_cams['Roll_Name'] = merged_cams['Name'].map(short_to_roll).fillna(merged_cams['Name'])
    merged_cams['Detection Count'] = pd.to_numeric(merged_cams['Detection Count'], errors='coerce').fillna(0)

    daily_counts = merged_cams.groupby(['Date', 'Roll_Name'])['Detection Count'].sum().reset_index()

    eval_records = []
    unique_dates = sorted(daily_counts['Date'].unique())
    all_students = sorted(list(roll_to_combo.keys()))

    for date_str in unique_dates:
        for roll in all_students:
            # Safely fetch true status to prevent missing data glitches
            if date_str in true_df.columns and roll in true_df.index:
                true_status_raw = str(true_df.at[roll, date_str]).strip().lower()
            else:
                true_status_raw = ""
                
            is_valid = true_status_raw not in ['nan', 'unknown', 'none', '']
            true_is_present = (true_status_raw == 'present')
            
            student_match = daily_counts[(daily_counts['Date'] == date_str) & (daily_counts['Roll_Name'] == roll)]
            count = student_match['Detection Count'].values[0] if not student_match.empty else 0

            eval_records.append({
                'Date': date_str,
                'Roll_Name': roll,
                'Total_Count': count,
                'True_Present': true_is_present,
                'True_Label': 'Present' if true_is_present else ('Absent' if is_valid else ''),
                'Is_Valid_True': is_valid
            })

    eval_df = pd.DataFrame(eval_records)
    if eval_df.empty:
        return pd.DataFrame(), {}

    # Calculate optimal threshold ignoring students with missing true attendance
    best_threshold = 1
    best_accuracy = 0.0
    valid_eval_df = eval_df[eval_df['Is_Valid_True']]
    
    if not valid_eval_df.empty:
        for t in range(1, 21):
            est_present = valid_eval_df['Total_Count'] >= t
            acc = (est_present == valid_eval_df['True_Present']).mean() * 100
            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = t

    tag = "A4 (Cam1 Only Daily)" if cam1_only else "A2 (All Cams Daily)"
    print(f"[{tag}] Optimal Threshold Found: >= {best_threshold} total detections/day (Overall Accuracy: {best_accuracy:.2f}%)")

    eval_df['Estimated'] = eval_df['Total_Count'].apply(lambda x: 'Present' if x >= best_threshold else 'Absent')

    idx = pd.Index(all_students, name='Roll_Name')
    combined_df = pd.DataFrame(index=idx)
    bottom_summary_row = {}

    label_col = ('Dataset', 'Label Size')
    combined_df[label_col] = [roll_to_labels.get(str(r), "N/A") for r in idx]
    valid_lbls = [v for v in combined_df[label_col] if isinstance(v, (int, float))]
    bottom_summary_row[label_col] = f"Avg: {sum(valid_lbls)/len(valid_lbls):.0f}" if valid_lbls else "N/A"

    total_matches_overall = 0
    total_possible_overall = 0

    for date_str in unique_dates:
        date_data = eval_df[eval_df['Date'] == date_str].set_index('Roll_Name')

        count_series = date_data['Total_Count']
        est_series = date_data['Estimated']
        true_series = date_data['True_Label']

        # only count matches where ground truth is valid
        valid_date_data = date_data[date_data['Is_Valid_True']]
        matches = (valid_date_data['Estimated'] == valid_date_data['True_Label']).sum()
        total = len(valid_date_data)

        if total > 0:
            date_eff = (matches / total) * 100
            total_matches_overall += matches
            total_possible_overall += total
            eff_str = f"{date_eff:.2f}%"
        else:
            eff_str = "N/A"

        bottom_summary_row[(date_str, 'Total Count')] = eff_str
        bottom_summary_row[(date_str, 'Estimated')] = ""
        bottom_summary_row[(date_str, 'True')] = ""

        combined_df[(date_str, 'Total Count')] = count_series.reindex(idx).fillna(0).astype(int)
        combined_df[(date_str, 'Estimated')] = est_series.reindex(idx).fillna("")
        combined_df[(date_str, 'True')] = true_series.reindex(idx).fillna("")

    student_matches = pd.Series(0.0, index=combined_df.index)
    student_valid = pd.Series(0.0, index=combined_df.index)

    for date_str in unique_dates:
        est_col = combined_df[(date_str, 'Estimated')]
        true_col = combined_df[(date_str, 'True')]

        # Prevent empty strings from breaking mask
        valid_mask = (~true_col.str.lower().isin(['unknown', 'nan', 'none', ''])) & (true_col != "")
        matches = (est_col == true_col) & valid_mask

        student_matches += matches.astype(int)
        student_valid += valid_mask.astype(int)

    student_acc = pd.Series(pd.NA, index=combined_df.index)
    mask = student_valid > 0
    student_acc[mask] = (student_matches[mask] / student_valid[mask]) * 100

    ov_col = ('Overall Summary', 'Student Accuracy')
    combined_df[ov_col] = student_acc.apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

    overall_eff_val = f"{(total_matches_overall / total_possible_overall) * 100:.2f}%" if total_possible_overall > 0 else "N/A"
    bottom_summary_row[ov_col] = overall_eff_val

    combined_df.columns = pd.MultiIndex.from_tuples(combined_df.columns)
    combined_df.index = combined_df.index.map(lambda x: roll_to_combo.get(str(x), str(x)))
    combined_df.index.name = 'Student Name & Roll No'
    combined_df = combined_df.sort_index()

    summary_df = pd.DataFrame([bottom_summary_row], index=['AVERAGE EFFICIENCY'])
    summary_df.columns = combined_df.columns
    combined_df = pd.concat([combined_df, summary_df])

    info = {
        'num_dates': len(unique_dates),
        'best_threshold': best_threshold,
        'overall_col': ov_col,
        'overall_eff_val': overall_eff_val
    }
    return combined_df, info


# ==============================================================================
# EXCEL FORMATTING FUNCTIONS
# ==============================================================================
def format_protocol_sheet(ws, num_sessions):
    ws['A1'] = 'Student Name & Roll No'
    ws.merge_cells('A1:A3')
    ws.delete_rows(4)

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
    thin_border = Border(
        left=Side(style='thin', color='BDC3C7'), right=Side(style='thin', color='BDC3C7'),
        top=Side(style='thin', color='BDC3C7'), bottom=Side(style='thin', color='BDC3C7')
    )

    for row in range(1, 4):
        for cell in ws[row]:
            try:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = center_align
                cell.border = thin_border
            except AttributeError:
                pass

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
                elif val.endswith('%') and val != 'n/a':
                    try:
                        if float(val.strip('%')) < 80.0:
                            cell.fill = red_fill
                            cell.font = red_font
                    except ValueError:
                        pass
            except (AttributeError, ValueError):
                pass

    for col_idx, cell in enumerate(ws[max_row], start=1):
        try:
            cell.alignment = center_align if col_idx > 1 else left_align
            cell.border = thin_border
            cell.fill = summary_fill
            cell.font = Font(bold=True)

            val = str(cell.value).strip().lower()
            if val.endswith('%') and val != 'n/a':
                try:
                    if float(val.strip('%')) < 80.0:
                        cell.fill = red_fill
                        cell.font = Font(color="9C0006", bold=True)
                except ValueError:
                    pass
        except (AttributeError, ValueError):
            pass

    for i in range(num_sessions):
        start_col = 3 + (i * 3)
        end_col = start_col + 2
        ws.merge_cells(start_row=max_row, start_column=start_col, end_row=max_row, end_column=end_col)

    for col_idx in range(1, max_col + 1):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = 15
    ws.column_dimensions['A'].width = 38
    ws.freeze_panes = "C4"


def format_daily_sheet(ws, num_dates, overall_eff_val):
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
    thin_border = Border(
        left=Side(style='thin', color='BDC3C7'), right=Side(style='thin', color='BDC3C7'),
        top=Side(style='thin', color='BDC3C7'), bottom=Side(style='thin', color='BDC3C7')
    )

    for row in range(1, 3):
        for cell in ws[row]:
            try:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = center_align
                cell.border = thin_border
            except AttributeError:
                pass

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
                    try:
                        if float(val.strip('%')) < 80.0:
                            cell.fill = red_fill
                            cell.font = red_font
                    except ValueError:
                        pass
            except (AttributeError, ValueError):
                pass

    for col_idx, cell in enumerate(ws[max_row], start=1):
        try:
            cell.alignment = center_align if col_idx > 1 else left_align
            cell.border = thin_border
            cell.fill = summary_fill
            cell.font = Font(bold=True)

            val = str(cell.value).strip().lower()
            if val.endswith('%') and val != 'n/a':
                try:
                    if float(val.strip('%')) < 80.0:
                        cell.fill = red_fill
                        cell.font = Font(color="9C0006", bold=True)
                except ValueError:
                    pass
        except (AttributeError, ValueError):
            pass

    for i in range(num_dates):
        start_col = 3 + (i * 3)
        end_col = start_col + 2
        ws.merge_cells(start_row=max_row, start_column=start_col, end_row=max_row, end_column=end_col)

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

    for col_idx in range(1, max_col + 1):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = 15 if col_idx < max_col else 20
    ws.column_dimensions['A'].width = 38
    ws.freeze_panes = "C3"


def format_summary_sheet(ws):
    ws['A1'] = 'Student Name & Roll No'
    ws.merge_cells('A1:A2')
    ws.delete_rows(3)

    max_row = ws.max_row
    max_col = ws.max_column

    header_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    red_font = Font(color="9C0006")
    summary_fill = PatternFill(start_color="D5F5E3", end_color="D5F5E3", fill_type="solid")
    center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left_align = Alignment(horizontal="left", vertical="center")
    thin_border = Border(
        left=Side(style='thin', color='BDC3C7'), right=Side(style='thin', color='BDC3C7'),
        top=Side(style='thin', color='BDC3C7'), bottom=Side(style='thin', color='BDC3C7')
    )

    for row in range(1, 3):
        for cell in ws[row]:
            try:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = center_align
                cell.border = thin_border
            except AttributeError:
                pass

    for row_idx in range(3, max_row):
        for col_idx, cell in enumerate(ws[row_idx], start=1):
            try:
                cell.alignment = left_align if col_idx == 1 else center_align
                cell.border = thin_border
                if row_idx % 2 == 0:
                    cell.fill = PatternFill(start_color="F8F9F9", end_color="F8F9F9", fill_type="solid")

                val = str(cell.value).strip().lower()
                if val.endswith('%') and val != 'n/a':
                    try:
                        if float(val.strip('%')) < 80.0:
                            cell.fill = red_fill
                            cell.font = red_font
                    except ValueError:
                        pass
            except (AttributeError, ValueError):
                pass

    for col_idx, cell in enumerate(ws[max_row], start=1):
        try:
            cell.alignment = center_align if col_idx > 1 else left_align
            cell.border = thin_border
            cell.fill = summary_fill
            cell.font = Font(bold=True)

            val = str(cell.value).strip().lower()
            if val.endswith('%') and val != 'n/a':
                try:
                    if float(val.strip('%')) < 80.0:
                        cell.fill = red_fill
                        cell.font = Font(color="9C0006", bold=True)
                except ValueError:
                    pass
        except (AttributeError, ValueError):
            pass

    for col_idx in range(1, max_col + 1):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = 16
    ws.column_dimensions['A'].width = 38
    ws.freeze_panes = "C3"


# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def process_all_attendance(base_dir, target_folder):
    input_dir = os.path.join(base_dir, target_folder)
    true_df, roll_to_combo, short_to_roll, base_to_protocol, roll_to_labels = load_reference_data(base_dir)

    file_pattern = os.path.join(input_dir, "*_output.csv")
    output_files = glob.glob(file_pattern)
    if not output_files:
        print(f"No output CSV files found in {input_dir}")
        return

    print("1. Running A1: Cam1 + Cam2 Union per Protocol...")
    df_a1, info_a1 = run_protocol_attendance(
        output_files, true_df, roll_to_combo, short_to_roll, base_to_protocol, roll_to_labels, cam1_only=False
    )

    print("2. Running A2: Cam1 + Cam2 Combined per Day with Threshold...")
    df_a2, info_a2 = run_daily_attendance(
        output_files, true_df, roll_to_combo, short_to_roll, roll_to_labels, cam1_only=False
    )

    print("3. Running A3: Cam1 Only per Protocol...")
    df_a3, info_a3 = run_protocol_attendance(
        output_files, true_df, roll_to_combo, short_to_roll, base_to_protocol, roll_to_labels, cam1_only=True
    )

    print("4. Running A4: Cam1 Only per Day with Threshold...")
    df_a4, info_a4 = run_daily_attendance(
        output_files, true_df, roll_to_combo, short_to_roll, roll_to_labels, cam1_only=True
    )

    summary_cols = {}
    if not df_a1.empty:
        summary_cols[('Dataset', 'Label Size')] = df_a1[('Dataset', 'Comparison', 'Label Size')]
        for prot, col_k in info_a1['protocol_cols']:
            summary_cols[('A1 (Cam1+Cam2 Protocol)', f'{prot} Protocol')] = df_a1[col_k]
        summary_cols[('A1 (Cam1+Cam2 Protocol)', 'Overall')] = df_a1[info_a1['overall_col']]

    if not df_a2.empty:
        thr2 = info_a2['best_threshold']
        summary_cols[('A2 (Cam1+Cam2 Daily)', f'Overall (Thr>={thr2})')] = df_a2[info_a2['overall_col']]

    if not df_a3.empty:
        for prot, col_k in info_a3['protocol_cols']:
            summary_cols[('A3 (Cam1 Only Protocol)', f'{prot} Protocol')] = df_a3[col_k]
        summary_cols[('A3 (Cam1 Only Protocol)', 'Overall')] = df_a3[info_a3['overall_col']]

    if not df_a4.empty:
        thr4 = info_a4['best_threshold']
        summary_cols[('A4 (Cam1 Only Daily)', f'Overall (Thr>={thr4})')] = df_a4[info_a4['overall_col']]

    summary_overview_df = pd.DataFrame(summary_cols)
    summary_overview_df.columns = pd.MultiIndex.from_tuples(summary_overview_df.columns)
    summary_overview_df.index.name = 'Student Name & Roll No'

    excel_output = os.path.join(input_dir, "All_4_Algorithms_Attendance_Report.xlsx")
    summary_csv_output = os.path.join(input_dir, "All_4_Algorithms_Summary.csv")

    summary_overview_df.to_csv(summary_csv_output)
    if not df_a1.empty:
        df_a1.to_csv(os.path.join(input_dir, "A1_Report_With_Efficiency.csv"))
    if not df_a2.empty:
        df_a2.to_csv(os.path.join(input_dir, "A2_Daily_Report_With_Efficiency.csv"))
    if not df_a3.empty:
        df_a3.to_csv(os.path.join(input_dir, "A3_Cam1_Report_With_Efficiency.csv"))
    if not df_a4.empty:
        df_a4.to_csv(os.path.join(input_dir, "A4_Cam1_Daily_Report_With_Efficiency.csv"))

    with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
        summary_overview_df.to_excel(writer, sheet_name='Summary_All_4_Algos')
        if not df_a1.empty:
            df_a1.to_excel(writer, sheet_name='A1_Protocol_Combined')
        if not df_a2.empty:
            df_a2.to_excel(writer, sheet_name='A2_Daily_Combined')
        if not df_a3.empty:
            df_a3.to_excel(writer, sheet_name='A3_Protocol_Cam1')
        if not df_a4.empty:
            df_a4.to_excel(writer, sheet_name='A4_Daily_Cam1')

    wb = openpyxl.load_workbook(excel_output)
    if 'Summary_All_4_Algos' in wb.sheetnames:
        format_summary_sheet(wb['Summary_All_4_Algos'])
    if 'A1_Protocol_Combined' in wb.sheetnames:
        format_protocol_sheet(wb['A1_Protocol_Combined'], info_a1['num_sessions'])
    if 'A2_Daily_Combined' in wb.sheetnames:
        format_daily_sheet(wb['A2_Daily_Combined'], info_a2['num_dates'], info_a2['overall_eff_val'])
    if 'A3_Protocol_Cam1' in wb.sheetnames:
        format_protocol_sheet(wb['A3_Protocol_Cam1'], info_a3['num_sessions'])
    if 'A4_Daily_Cam1' in wb.sheetnames:
        format_daily_sheet(wb['A4_Daily_Cam1'], info_a4['num_dates'], info_a4['overall_eff_val'])

    wb.save(excel_output)

    print("\nSuccess! Generated unified Excel sheet and CSV files:")
    print(f" -> Excel Workbook : {excel_output}")
    print(f" -> Summary CSV    : {summary_csv_output}")


if __name__ == "__main__":
    BASE_DIRECTORY = "ATTENDENCE RESULTS/Results"
    TARGET_FOLDER = "test 10"

    process_all_attendance(BASE_DIRECTORY, TARGET_FOLDER)