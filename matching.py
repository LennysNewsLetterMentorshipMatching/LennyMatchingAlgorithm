"""
MENTOR-MENTEE MATCHING ALGORITHM - COMPATIBILITY-FIXED VERSION
====
This version completely avoids PyTorch/transformers dependencies that cause
the 'torch.compiler' AttributeError. Uses pure text similarity instead.
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
from ortools.linear_solver import pywraplp
import re

# ================
# CONFIGURATION & FILE HANDLING  
# ================

MENTORS_CSV = 'Mentor-Original-Submissions.csv'
MENTEES_CSV = 'Mentee-Original-Submissions.csv'
OUTPUT_CSV = 'mentor_then_mentees_matches_fixed.csv'

for filepath in [MENTORS_CSV, MENTEES_CSV]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

# ================
# DATA LOADING & PREPROCESSING
# ================

print("Loading mentor and mentee data...")
mentors = pd.read_csv(MENTORS_CSV)
mentees = pd.read_csv(MENTEES_CSV)

# Handle timezone column differences
if "Timezones" in mentors.columns:
    mentors["Timezone"] = mentors["Timezones"]
elif "Timezone" not in mentors.columns:
    raise ValueError("Neither 'Timezone' nor 'Timezones' column found in mentors CSV")

if "Timezone" not in mentees.columns:
    raise ValueError("'Timezone' column not found in mentees CSV")

print(f"✅ Standardized timezone columns")

TIMEZONE_OFFSETS = {
    'PST - Pacific US/Canada': -8, 'EST - Eastern US/Canada': -5, 
    'CST - Central US/Canada': -6, 'MST - Mountain US/Canada': -7,
    'GMT - Greenwich Mean Time': 0, 'CET - Central Europe': 1, 
    'EET - Eastern Europe': 2, 'IST - India': 5.5, 'JST - Japan': 9,
    'AEST - Eastern Australia': 10, 'SGT - Singapore': 8, 
    'WET - Western Europe': 0, 'BRT - Brazil (Brasília)': -3,
    'CST - China': 8, 'ICT - Thailand': 7, 'TPE - Taipei': 8,
    'MSK - Moscow, Russia': 3, 'AST - Atlantic Canada': -4
}

def parse_timezone_to_offset(timezone_str):
    if pd.isna(timezone_str):
        return np.nan
    timezone_str = str(timezone_str).strip()
    if timezone_str in TIMEZONE_OFFSETS:
        return TIMEZONE_OFFSETS[timezone_str]
    try:
        return float(timezone_str)
    except:
        return 0

def parse_comma_separated_to_set(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return set(item.strip() for item in str(value).split(',') if item.strip())

def parse_years_of_experience(value):
    if pd.isna(value):
        return np.nan
    value_str = str(value).strip()
    if '-' in value_str:
        try:
            parts = value_str.split('-')
            if len(parts) == 2:
                return (float(parts[0]) + float(parts[1])) / 2
        except:
            pass
    if '+' in value_str:
        try:
            return float(value_str.replace('+', ''))
        except:
            pass
    try:
        return float(value_str)
    except:
        return np.nan

def parse_previous_matches(value):
    if pd.isna(value):
        return False
    value_str = str(value).strip().lower()
    has_previous_indicators = ['yes', 'yes - as a mentee', 'previously matched']
    no_previous_indicators = ['no', 'yes - as a mentor', 'first time']
    
    for indicator in has_previous_indicators:
        if indicator in value_str:
            return True
    return False

# Apply preprocessing
mentors["Offset"] = mentors["Timezone"].apply(parse_timezone_to_offset)
mentees["Offset"] = mentees["Timezone"].apply(parse_timezone_to_offset)
mentors["Years of Experience"] = mentors["Years of Experience"].apply(parse_years_of_experience)
mentees["Years of Experience"] = mentees["Years of Experience"].apply(parse_years_of_experience)
mentees["Has_Previous_Matches"] = mentees["Previous Matches"].apply(parse_previous_matches)

mentors["Full Name"] = mentors["First Name"].astype(str) + " " + mentors["Last Name"].astype(str)
mentees["Full Name"] = mentees["First Name"].astype(str) + " " + mentees["Last Name"].astype(str)

multi_select_columns = ["Roles", "Topics", "Industry", "Company Stage", "In-Person Meeting Location"]
for column in multi_select_columns:
    mentors[column] = mentors[column].apply(parse_comma_separated_to_set)
    mentees[column] = mentees[column].apply(parse_comma_separated_to_set)

important_attribute_columns = [
    "Important Attribute - First", "Important Attribute - Second", "Important Attribute - Third"
]

for column in important_attribute_columns:
    mentors[column] = mentors[column].fillna("No Preference").astype(str).str.strip()
    mentees[column] = mentees[column].fillna("No Preference").astype(str).str.strip()

mentors["Time Slot Preference"] = mentors["Time Slot Preference"].fillna("No Preference").astype(str).str.strip()
mentees["Time Slot Preference"] = mentees["Time Slot Preference"].fillna("No Preference").astype(str).str.strip()

new_mentees_count = (~mentees["Has_Previous_Matches"]).sum()
returning_mentees_count = mentees["Has_Previous_Matches"].sum()
print(f"📊 Mentee Status: {new_mentees_count} new mentees, {returning_mentees_count} returning mentees")

# ================
# ENHANCED TEXT-BASED GOAL ANALYSIS
# ================

def extract_keywords(text):
    """Extract meaningful keywords from text for matching"""
    if pd.isna(text) or str(text).strip() == "":
        return set()
    
    text = str(text).lower()
    
    # Remove punctuation and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    
    # Expanded stop words list
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they', 'them', 'their',
        'this', 'that', 'these', 'those', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 
        'might', 'can', 'get', 'got', 'make', 'take', 'see', 'know', 'think', 'want', 'need',
        'like', 'work', 'time', 'way', 'look', 'new', 'good', 'great', 'help', 'also', 'well',
        'more', 'very', 'much', 'some', 'any', 'all', 'most', 'other', 'such', 'even', 'just',
        'than', 'only', 'over', 'back', 'after', 'use', 'two', 'how', 'its', 'our', 'out', 'day',
        'find', 'give', 'man', 'here', 'old', 'life', 'own', 'say', 'her', 'would', 'make',
        'him', 'into', 'more', 'go', 'no', 'so', 'what', 'up', 'year', 'about', 'come', 'could',
        'now', 'than', 'them', 'these', 'want', 'way', 'who', 'boy', 'did', 'its', 'let', 'put',
        'too', 'try'
    }
    
    # Filter out stop words and keep meaningful terms
    meaningful_words = set(word for word in words if word not in stop_words and len(word) > 3)
    
    return meaningful_words

def calculate_enhanced_text_similarity(text1, text2):
    """Enhanced text similarity using keyword extraction and weighted matching"""
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)
    
    if len(keywords1) == 0 or len(keywords2) == 0:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(keywords1 & keywords2)
    union = len(keywords1 | keywords2)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Calculate weighted overlap (favor longer shared keywords)
    shared_keywords = keywords1 & keywords2
    weight_score = sum(len(word) for word in shared_keywords) / max(
        sum(len(word) for word in keywords1), 
        sum(len(word) for word in keywords2), 
        1
    )
    
    # Combine scores
    final_similarity = (jaccard * 0.7) + (weight_score * 0.3)
    return min(final_similarity, 1.0)

# Extract and process goal texts
mentor_goal_texts = mentors["Open Answer"].fillna("").tolist()
mentee_goal_texts = mentees["Open Answer"].fillna("").tolist()

# Create topic assignments based on keyword clustering
def simple_topic_assignment(texts):
    """Assign simple topic categories based on common keywords"""
    topics = []
    
    for text in texts:
        keywords = extract_keywords(text)
        
        # Define topic categories
        if any(word in keywords for word in ['leadership', 'manage', 'team', 'lead', 'director']):
            topics.append(1)  # Leadership
        elif any(word in keywords for word in ['technical', 'engineering', 'developer', 'code', 'software']):
            topics.append(2)  # Technical
        elif any(word in keywords for word in ['product', 'strategy', 'market', 'business']):
            topics.append(3)  # Product/Strategy
        elif any(word in keywords for word in ['startup', 'entrepreneur', 'founder', 'venture']):
            topics.append(4)  # Startup
        elif any(word in keywords for word in ['career', 'transition', 'change', 'move', 'switch']):
            topics.append(5)  # Career Transition
        else:
            topics.append(0)  # General
    
    return topics

mentors["Goal_Topic"] = simple_topic_assignment(mentor_goal_texts)
mentees["Goal_Topic"] = simple_topic_assignment(mentee_goal_texts)

print("✅ Using enhanced text-based goal matching (no PyTorch dependencies)")

# ================
# SCORING SYSTEM
# ================

BASE_SCORE = 0.0

def calculate_new_mentee_bonus(has_previous_matches):
    return 50.0 if not has_previous_matches else 0.0

def calculate_goal_alignment_bonus(mentor_idx, mentee_idx):
    """Enhanced goal alignment using text similarity"""
    mentor_text = mentor_goal_texts[mentor_idx]
    mentee_text = mentee_goal_texts[mentee_idx]
    similarity = calculate_enhanced_text_similarity(mentor_text, mentee_text)
    return similarity * 75  # Scale to match expected range

def calculate_topic_alignment_bonus(mentor_topic, mentee_topic):
    if mentor_topic == mentee_topic and mentor_topic != 0:  # Don't reward generic matches
        return 30.0
    return 0.0

def calculate_experience_bonus(mentor_yoe, mentee_yoe):
    if pd.isna(mentor_yoe) or pd.isna(mentee_yoe):
        mentor_yoe = mentor_yoe if not pd.isna(mentor_yoe) else 5.0
        mentee_yoe = mentee_yoe if not pd.isna(mentee_yoe) else 2.0
    
    experience_gap = mentor_yoe - mentee_yoe
    if experience_gap <= 0:
        return -np.inf
    
    if experience_gap >= 8: return 160.0
    elif experience_gap >= 4: return 120.0
    elif experience_gap >= 2: return 100.0
    elif experience_gap >= 1: return 80.0
    else: return 50.0

def calculate_timezone_flexibility_bonus(mentor_offset, mentee_offset):
    if pd.isna(mentor_offset) or pd.isna(mentee_offset):
        return 10.0
    
    time_difference = abs(mentor_offset - mentee_offset)
    if time_difference <= 1: return 30.0
    elif time_difference <= 2: return 20.0
    elif time_difference <= 4: return 5.0
    elif time_difference <= 6: return -10.0
    elif time_difference <= 8: return -25.0
    else: return -40.0

def calculate_time_slot_bonus(mentor_slot, mentee_slot):
    if mentor_slot == "No Preference" or mentee_slot == "No Preference":
        return 0.0
    return 15.0 if mentor_slot == mentee_slot else 0.0

def calculate_attribute_match_points(mentor_set, mentee_set, base_points=10, exclude_no_preference=False):
    if exclude_no_preference:
        mentor_filtered = {item for item in mentor_set if item.lower() != 'no preference'}
        mentee_filtered = {item for item in mentee_set if item.lower() != 'no preference'}
        shared_items = mentor_filtered & mentee_filtered
    else:
        shared_items = mentor_set & mentee_set
    return len(shared_items) * base_points

def compute_overall_match_score(mentor_row, mentee_row, mentor_idx, mentee_idx):
    experience_bonus = calculate_experience_bonus(
        mentor_row["Years of Experience"], mentee_row["Years of Experience"]
    )
    if experience_bonus == -np.inf:
        return -np.inf
    
    total_score = BASE_SCORE
    total_score += calculate_new_mentee_bonus(mentee_row["Has_Previous_Matches"])
    total_score += calculate_goal_alignment_bonus(mentor_idx, mentee_idx)
    total_score += calculate_topic_alignment_bonus(mentor_row["Goal_Topic"], mentee_row["Goal_Topic"])
    total_score += experience_bonus
    total_score += calculate_timezone_flexibility_bonus(mentor_row["Offset"], mentee_row["Offset"])
    
    # Attribute bonuses
    total_score += calculate_attribute_match_points(mentor_row["Roles"], mentee_row["Roles"], 15)
    total_score += calculate_attribute_match_points(mentor_row["Topics"], mentee_row["Topics"], 12)
    total_score += calculate_attribute_match_points(mentor_row["Industry"], mentee_row["Industry"], 10)
    total_score += calculate_attribute_match_points(mentor_row["Company Stage"], mentee_row["Company Stage"], 10, exclude_no_preference=True)
    total_score += calculate_attribute_match_points(mentor_row["In-Person Meeting Location"], mentee_row["In-Person Meeting Location"], 5)
    
    # Important attributes bonus
    for attribute_column in important_attribute_columns:
        mentor_value = mentor_row.get(attribute_column, "No Preference")
        mentee_value = mentee_row.get(attribute_column, "No Preference")
        if (mentor_value != "No Preference" and mentee_value != "No Preference" and mentor_value == mentee_value):
            total_score += 15.0
    
    total_score += calculate_time_slot_bonus(mentor_row["Time Slot Preference"], mentee_row["Time Slot Preference"])
    
    return max(total_score, 50)

# ================
# OPTIMIZATION
# ================

print("Computing compatibility scores...")
mentor_indices = mentors.index.tolist()
mentee_indices = mentees.index.tolist()
pair_compatibility_scores = {}

for mentor_idx in mentor_indices:
    for mentee_idx in mentee_indices:
        score = compute_overall_match_score(
            mentors.loc[mentor_idx], mentees.loc[mentee_idx], mentor_idx, mentee_idx
        )
        if score > -np.inf:
            pair_compatibility_scores[(mentor_idx, mentee_idx)] = score

print(f"Found {len(pair_compatibility_scores)} viable pairs")

solver = pywraplp.Solver.CreateSolver('SCIP')
matching_variables = {}
for (mentor_idx, mentee_idx) in pair_compatibility_scores.keys():
    matching_variables[(mentor_idx, mentee_idx)] = solver.BoolVar(f"x_{mentor_idx}_{mentee_idx}")

# Constraints
for mentee_idx in mentee_indices:
    potential_matches = [matching_variables[(mentor_idx, mentee_idx)] 
                        for mentor_idx in mentor_indices 
                        if (mentor_idx, mentee_idx) in matching_variables]
    if potential_matches:
        solver.Add(solver.Sum(potential_matches) <= 1)

for mentor_idx in mentor_indices:
    potential_matches = [matching_variables[(mentor_idx, mentee_idx)] 
                        for mentee_idx in mentee_indices 
                        if (mentor_idx, mentee_idx) in matching_variables]
    if potential_matches:
        solver.Add(solver.Sum(potential_matches) <= 2)

# Objective
total_matches = solver.Sum([matching_variables[(mentor_idx, mentee_idx)] 
                           for (mentor_idx, mentee_idx) in matching_variables.keys()])
total_quality = solver.Sum([score * matching_variables[(mentor_idx, mentee_idx)]
                           for (mentor_idx, mentee_idx), score in pair_compatibility_scores.items()])

solver.Maximize(1000 * total_matches + total_quality)

print("Solving optimization problem...")
solution_status = solver.Solve()

if solution_status != pywraplp.Solver.OPTIMAL:
    raise RuntimeError("No optimal solution found.")

# Extract results
mentor_to_mentees_mapping = defaultdict(list)
for (mentor_idx, mentee_idx), var in matching_variables.items():
    if var.solution_value() > 0.5:
        mentor_to_mentees_mapping[mentor_idx].append(mentee_idx)

total_matched = sum(len(mentees_list) for mentees_list in mentor_to_mentees_mapping.values())
matched_new = sum(1 for mentor_idx, mentee_list in mentor_to_mentees_mapping.items()
                 for mentee_idx in mentee_list
                 if not mentees.loc[mentee_idx, "Has_Previous_Matches"])

print(f"✅ Matched {total_matched}/{len(mentees)} mentees ({total_matched/len(mentees)*100:.1f}%)")
print(f"🆕 New mentees matched: {matched_new}")

# ================
# OUTPUT
# ================

output_columns = ["Offset", "Years of Experience", "Roles", "Topics", "Industry", 
                 "Company Stage", "In-Person Meeting Location", "Time Slot Preference", 
                 "Goal_Topic", "Has_Previous_Matches"]

def format_set_for_output(value):
    return "; ".join(sorted(value)) if isinstance(value, set) else value

output_data_rows = []
used_submission_ids = set()

for mentor_idx in sorted(mentor_to_mentees_mapping.keys(), key=lambda x: mentors.loc[x, "Full Name"]):
    mentor_data = mentors.loc[mentor_idx]
    mentor_submission_id = mentor_data.get("Mentor Submission ID", "")
    
    if mentor_submission_id and mentor_submission_id not in used_submission_ids:
        used_submission_ids.add(mentor_submission_id)
        
        # Add mentor row
        mentor_row = {
            "Row Type": "Mentor", "Name": mentor_data.get("Full Name", ""),
            "Email": mentor_data.get("Coda Email", ""), "Submission ID": mentor_submission_id,
            "Match Score": ""
        }
        for column in output_columns:
            if column in mentor_data.index:
                mentor_row[column] = format_set_for_output(mentor_data[column])
            else:
                mentor_row[column] = "N/A"
        output_data_rows.append(mentor_row)
        
        # Add mentee rows
        assigned_mentees = sorted(mentor_to_mentees_mapping[mentor_idx], 
                                key=lambda x: -pair_compatibility_scores[(mentor_idx, x)])
        
        for mentee_idx in assigned_mentees:
            mentee_data = mentees.loc[mentee_idx]
            mentee_submission_id = mentee_data.get("Mentee Submission ID", "")
            
            if mentee_submission_id and mentee_submission_id not in used_submission_ids:
                used_submission_ids.add(mentee_submission_id)
                match_score = pair_compatibility_scores[(mentor_idx, mentee_idx)]
                
                mentee_row = {
                    "Row Type": "Mentee", "Name": mentee_data.get("Full Name", ""),
                    "Email": mentee_data.get("Coda Email", ""), "Submission ID": mentee_submission_id,
                    "Match Score": round(match_score, 2)
                }
                for column in output_columns:
                    mentee_row[column] = format_set_for_output(mentee_data[column])
                output_data_rows.append(mentee_row)

# Save results
final_output_df = pd.DataFrame(output_data_rows)
final_output_df = final_output_df.drop_duplicates(subset=['Submission ID'], keep='first')
final_output_df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Results saved to: {OUTPUT_CSV}")
print("\n" + "="*50)
print("SCORING BREAKDOWN")
print("="*50)
print("🆕 NEW MENTEE BONUS: +50 points")
print("🎯 Goal Alignment: Up to 75 points (Enhanced text similarity)") 
print("🎯 Topic Match Bonus: Up to 30 points")
print("💼 Experience Gap: 50-160 points")
print("🌍 Timezone Flexibility: -40 to +30 points")
print("⏰ Time Slot Match: Up to 15 points")
print("🏢 Company Stage: Excludes 'No Preference'")
print("📊 Other Attributes: Various points")
print("="*50)
print("🔧 FIXED: No PyTorch/transformers dependencies - pure text matching")
print("="*50)
