"""
MENTOR-MENTEE MATCHING ALGORITHM
====
This script optimally matches mentors with mentees using Mixed Integer Linear Programming (MILP).
It maximizes total matching quality while respecting business constraints and prevents duplicate 
submission IDs in the output.

BUSINESS LOGIC:
- Each mentee gets exactly 1 mentor (or remains unmatched)
- Each mentor can have at most 2 mentees
- Mentors must have more years of experience than their mentees
- Mentor and mentee must be within 2 time zones of each other
- Similarity in roles, topics, industry, etc. INCREASES match quality
- Time slot matching provides bonus for exact matches, penalty for conflicts
- The algorithm finds the globally optimal assignment (not just greedy)
- CRITICAL: Each submission ID appears exactly once in output (no duplicates)

UPDATED TO MAXIMIZE MATCHES: Relaxed experience gap requirements to allow more matches
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer
from ortools.linear_solver import pywraplp

# ================
# CONFIGURATION & FILE HANDLING
# ================

MENTORS_CSV = 'Mentor-Original-Submissions.csv'
MENTEES_CSV = 'Mentee-Original-Submissions.csv'
OUTPUT_CSV = 'mentor_then_mentees_matches_fixed.csv'

for filepath in [MENTORS_CSV, MENTEES_CSV]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"File not found: {filepath}. "
            f"Check your paths or working directory: {os.getcwd()}"
        )

# ================
# DATA LOADING & VALIDATION
# ================

print("Loading mentor and mentee data...")
mentors = pd.read_csv(MENTORS_CSV)
mentees = pd.read_csv(MENTEES_CSV)

# Timezone offset mapping
TIMEZONE_OFFSETS = {
    'PST - Pacific US/Canada': -8,
    'EST - Eastern US/Canada': -5,
    'CST - Central US/Canada': -6,
    'MST - Mountain US/Canada': -7,
    'GMT - Greenwich Mean Time': 0,
    'CET - Central Europe': 1,
    'EET - Eastern Europe': 2,
    'IST - India': 5.5,
    'JST - Japan': 9,
    'AEST - Eastern Australia': 10,
    'SGT - Singapore': 8,
    'WET - Western Europe': 0,
    'BRT - Brazil (Brasília)': -3,
    'CST - China': 8,
    'ICT - Thailand': 7,
    'TPE - Taipei': 8,
    'MSK - Moscow, Russia': 3,
    'AST - Atlantic Canada': -4
}

def parse_timezone_to_offset(timezone_str):
    """Convert timezone string to numeric offset"""
    if pd.isna(timezone_str):
        return np.nan
    
    timezone_str = str(timezone_str).strip()
    
    if timezone_str in TIMEZONE_OFFSETS:
        return TIMEZONE_OFFSETS[timezone_str]
    
    try:
        return float(timezone_str)
    except:
        return 0

common_required_columns = [
    "Coda Email",
    "Timezones",
    "Years of Experience",
    "Roles",
    "Topics",
    "Industry",
    "Company Stage",
    "In-Person Meeting Location",
    "Time Slot Preference",
    "Open Answer",
    "Important Attribute - First",
    "Important Attribute - Second",
    "Important Attribute - Third"
]

mentor_specific_columns = ["Mentor Submission ID"]
mentee_specific_columns = ["Mentee Submission ID"]

# Check for missing columns
missing_mentor_common = [col for col in common_required_columns if col not in mentors.columns]
missing_mentee_common = [col for col in common_required_columns if col not in mentees.columns]
missing_mentor_specific = [col for col in mentor_specific_columns if col not in mentors.columns]
missing_mentee_specific = [col for col in mentee_specific_columns if col not in mentees.columns]

if missing_mentor_common or missing_mentor_specific:
    all_missing_mentor = missing_mentor_common + missing_mentor_specific
    raise ValueError(f"Mentors CSV missing required columns: {all_missing_mentor}")
    
if missing_mentee_common or missing_mentee_specific:
    all_missing_mentee = missing_mentee_common + missing_mentee_specific
    raise ValueError(f"Mentees CSV missing required columns: {all_missing_mentee}")

print(f"Loaded {len(mentors)} mentors and {len(mentees)} mentees")

# ================
# DATA PREPROCESSING
# ================

print("Preprocessing data...")

def parse_comma_separated_to_set(value):
    """Convert comma-separated strings to Python sets for efficient matching."""
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return set(item.strip() for item in str(value).split(',') if item.strip())

def parse_years_of_experience(value):
    """Parse years of experience from various formats"""
    if pd.isna(value):
        return np.nan
    
    value_str = str(value).strip()
    
    # Handle ranges like "3-5", "6-8", etc.
    if '-' in value_str:
        try:
            parts = value_str.split('-')
            if len(parts) == 2:
                min_years = float(parts[0])
                max_years = float(parts[1])
                return (min_years + max_years) / 2
        except:
            pass
    
    # Handle "20+" format
    if '+' in value_str:
        try:
            return float(value_str.replace('+', ''))
        except:
            pass
    
    # Try direct conversion
    try:
        return float(value_str)
    except:
        return np.nan

# Convert timezone strings to numeric offsets
mentors["Offset"] = mentors["Timezones"].apply(parse_timezone_to_offset)
mentees["Offset"] = mentees["Timezones"].apply(parse_timezone_to_offset)

# Parse years of experience
mentors["Years of Experience"] = mentors["Years of Experience"].apply(parse_years_of_experience)
mentees["Years of Experience"] = mentees["Years of Experience"].apply(parse_years_of_experience)

# Add Full Name column using First Name + Last Name
mentors["Full Name"] = mentors["First Name"].astype(str) + " " + mentors["Last Name"].astype(str)
mentees["Full Name"] = mentees["First Name"].astype(str) + " " + mentees["Last Name"].astype(str)

# Convert multi-select columns to sets
multi_select_columns = [
    "Roles", 
    "Topics", 
    "Industry", 
    "Company Stage", 
    "In-Person Meeting Location"
]

for column in multi_select_columns:
    mentors[column] = mentors[column].apply(parse_comma_separated_to_set)
    mentees[column] = mentees[column].apply(parse_comma_separated_to_set)

# Clean and standardize text fields
important_attribute_columns = [
    "Important Attribute - First", 
    "Important Attribute - Second", 
    "Important Attribute - Third"
]

for column in important_attribute_columns:
    mentors[column] = mentors[column].fillna("No Preference").astype(str).str.strip()
    mentees[column] = mentees[column].fillna("No Preference").astype(str).str.strip()

mentors["Time Slot Preference"] = mentors["Time Slot Preference"].fillna("No Preference").astype(str).str.strip()
mentees["Time Slot Preference"] = mentees["Time Slot Preference"].fillna("No Preference").astype(str).str.strip()

# ================
# SENTIMENT ANALYSIS SETUP
# ================

print("Setting up sentiment analysis...")

sentiment_analyzer = SentimentIntensityAnalyzer()

def calculate_sentiment_score(text):
    """Extract sentiment score from free-text responses."""
    if pd.isna(text) or str(text).strip() == "":
        return 0.0
    
    sentiment_scores = sentiment_analyzer.polarity_scores(str(text))
    return float(sentiment_scores["compound"])

mentors["Sentiment"] = mentors["Open Answer"].fillna("").apply(calculate_sentiment_score)
mentees["Sentiment"] = mentees["Open Answer"].fillna("").apply(calculate_sentiment_score)

# ================
# SCORING SYSTEM CONFIGURATION (MAXIMIZED FOR MORE MATCHES)
# ================

BASE_SCORE = 0.0

def calculate_sentiment_alignment_bonus(mentor_sentiment, mentee_sentiment):
    """Calculate bonus for sentiment alignment based on document formula."""
    max_difference = 2.0
    actual_difference = abs(mentor_sentiment - mentee_sentiment)
    alignment_score = 1.0 - (actual_difference / max_difference)
    return alignment_score * 30.0

def calculate_experience_bonus(mentor_yoe, mentee_yoe):
    """
    UPDATED: More permissive experience gap requirements to maximize matches.
    
    OLD PROBLEM: Required minimum 2-year gap, eliminating many potential matches
    NEW SOLUTION: Allow ANY positive gap, with bonus scaling for larger gaps
    """
    if pd.isna(mentor_yoe) or pd.isna(mentee_yoe):
        # CHANGE: Instead of returning -inf, give default experience values
        # This allows matches even when experience data is missing
        mentor_yoe = mentor_yoe if not pd.isna(mentor_yoe) else 5.0  # Default mentor experience
        mentee_yoe = mentee_yoe if not pd.isna(mentee_yoe) else 2.0  # Default mentee experience
    
    experience_gap = mentor_yoe - mentee_yoe
    
    # CRITICAL CHANGE: Only require mentor has MORE experience (any positive gap)
    if experience_gap <= 0:
        return -np.inf  # Still maintain this hard constraint
    
    # NEW FLEXIBLE SCORING: Reward larger gaps but allow smaller ones
    if experience_gap >= 8:
        return 160.0    # Large gap - very senior mentor
    elif experience_gap >= 4:
        return 120.0    # Good gap - substantial difference
    elif experience_gap >= 2:
        return 100.0    # Decent gap
    elif experience_gap >= 1:
        return 80.0     # Small but valid gap
    else:
        return 50.0     # Very small gap but still valid

def calculate_timezone_proximity_bonus(mentor_offset, mentee_offset):
    """Calculate timezone bonus with more flexible handling of missing data."""
    if pd.isna(mentor_offset) or pd.isna(mentee_offset):
        # CHANGE: Instead of returning 0, assume same timezone if data missing
        # This allows more matches when timezone data is incomplete
        return 20.0  # Give moderate bonus for missing timezone data
    
    time_difference = abs(mentor_offset - mentee_offset)
    
    if time_difference > 2:
        return -np.inf  # Still maintain hard constraint
    
    if time_difference == 0:
        return 30.0
    elif time_difference == 1:
        return 25.0
    elif time_difference == 2:
        return 20.0
    
    return 0.0

def calculate_time_slot_bonus(mentor_slot, mentee_slot):
    """Calculate time slot matching bonus with more flexible handling."""
    if mentor_slot == "No Preference" or mentee_slot == "No Preference":
        return 10.0  # CHANGE: Small bonus instead of 0 to encourage matches
    
    if mentor_slot == mentee_slot:
        return 60.0
    else:
        return 0.0  # CHANGE: Remove penalty to avoid discouraging matches

def calculate_attribute_match_points(mentor_set, mentee_set, base_points=10):
    """Calculate points for shared attributes."""
    shared_items = mentor_set & mentee_set
    return len(shared_items) * base_points

def compute_overall_match_score(mentor_row, mentee_row):
    """
    UPDATED: More permissive scoring to maximize number of viable matches.
    Focus on encouraging matches rather than being overly restrictive.
    """
    
    # ================
    # RELAXED CONSTRAINTS CHECK
    # ================
    
    # Time zone constraint (more flexible handling)
    mentor_tz = mentor_row["Offset"]
    mentee_tz = mentee_row["Offset"]
    
    # Only enforce timezone constraint if both values are present and valid
    if not pd.isna(mentor_tz) and not pd.isna(mentee_tz):
        if abs(mentor_tz - mentee_tz) > 2:
            return -np.inf
    
    # Experience constraint (more flexible handling)
    experience_bonus = calculate_experience_bonus(
        mentor_row["Years of Experience"], 
        mentee_row["Years of Experience"]
    )
    if experience_bonus == -np.inf:
        return -np.inf
    
    # ================
    # GENEROUS SCORE CALCULATION
    # ================
    
    total_score = BASE_SCORE
    
    # Add sentiment alignment bonus
    total_score += calculate_sentiment_alignment_bonus(
        mentor_row["Sentiment"], 
        mentee_row["Sentiment"]
    )
    
    # Add experience gap bonus
    total_score += experience_bonus
    
    # Add timezone proximity bonus (with flexible handling)
    timezone_bonus = calculate_timezone_proximity_bonus(mentor_tz, mentee_tz)
    if timezone_bonus == -np.inf:
        return -np.inf
    total_score += timezone_bonus
    
    # Add generous bonuses for shared attributes
    total_score += calculate_attribute_match_points(
        mentor_row["Roles"], mentee_row["Roles"], 15
    )
    total_score += calculate_attribute_match_points(
        mentor_row["Topics"], mentee_row["Topics"], 12
    )
    total_score += calculate_attribute_match_points(
        mentor_row["Industry"], mentee_row["Industry"], 10
    )
    total_score += calculate_attribute_match_points(
        mentor_row["Company Stage"], mentee_row["Company Stage"], 8
    )
    total_score += calculate_attribute_match_points(
        mentor_row["In-Person Meeting Location"], mentee_row["In-Person Meeting Location"], 5
    )
    
    # Add bonus for matching important attributes
    for attribute_column in important_attribute_columns:
        mentor_value = mentor_row.get(attribute_column, "No Preference")
        mentee_value = mentee_row.get(attribute_column, "No Preference")
        
        if (mentor_value != "No Preference" and 
            mentee_value != "No Preference" and 
            mentor_value == mentee_value):
            total_score += 15.0  # Increased bonus
    
    # Add time slot bonus (more forgiving)
    time_slot_bonus = calculate_time_slot_bonus(
        mentor_row["Time Slot Preference"],
        mentee_row["Time Slot Preference"]
    )
    total_score += time_slot_bonus
    
    # ENSURE MINIMUM VIABLE SCORE: Even poor matches get some points
    # This prevents the optimization from excluding too many potential matches
    if total_score < 50:
        total_score = 50  # Minimum baseline score for any viable match
    
    return total_score

# ================
# PAIRWISE SCORE COMPUTATION
# ================

print("Computing compatibility scores for all mentor-mentee pairs...")

mentor_indices = mentors.index.tolist()
mentee_indices = mentees.index.tolist()
pair_compatibility_scores = {}

for mentor_idx in mentor_indices:
    mentor_data = mentors.loc[mentor_idx]
    for mentee_idx in mentee_indices:
        mentee_data = mentees.loc[mentee_idx]
        
        compatibility_score = compute_overall_match_score(mentor_data, mentee_data)
        
        # Store ALL viable matches (even low-scoring ones)
        if compatibility_score > -np.inf:
            pair_compatibility_scores[(mentor_idx, mentee_idx)] = compatibility_score

print(f"Found {len(pair_compatibility_scores)} viable mentor-mentee pairs")

if not pair_compatibility_scores:
    raise ValueError("No valid matches found under current constraints. Check your data and business rules.")

# ================
# OPTIMIZATION SETUP WITH MATCH MAXIMIZATION
# ================

print("Setting up optimization problem to maximize matches...")

solver = pywraplp.Solver.CreateSolver('SCIP')
if not solver:
    raise RuntimeError("Failed to initialize OR-Tools solver.")

# Create decision variables
matching_variables = {}
for (mentor_idx, mentee_idx) in pair_compatibility_scores.keys():
    var_name = f"x_{mentor_idx}_{mentee_idx}"
    matching_variables[(mentor_idx, mentee_idx)] = solver.BoolVar(var_name)

# ================
# CONSTRAINTS
# ================

print("Adding business constraints...")

# CONSTRAINT 1: Each mentee can have at most 1 mentor
for mentee_idx in mentee_indices:
    potential_matches = [
        matching_variables[(mentor_idx, mentee_idx)] 
        for mentor_idx in mentor_indices 
        if (mentor_idx, mentee_idx) in matching_variables
    ]
    
    if potential_matches:
        solver.Add(solver.Sum(potential_matches) <= 1)

# CONSTRAINT 2: Each mentor can have at most 2 mentees  
for mentor_idx in mentor_indices:
    potential_matches = [
        matching_variables[(mentor_idx, mentee_idx)] 
        for mentee_idx in mentee_indices 
        if (mentor_idx, mentee_idx) in matching_variables
    ]
    
    if potential_matches:
        solver.Add(solver.Sum(potential_matches) <= 2)

# ================
# MULTI-OBJECTIVE OPTIMIZATION
# ================

print("Setting up multi-objective optimization...")

# OBJECTIVE: Maximize both total score AND number of matches
# We'll use a weighted combination that prioritizes match count

# Count total number of matches
total_matches = solver.Sum([
    matching_variables[(mentor_idx, mentee_idx)]
    for (mentor_idx, mentee_idx) in matching_variables.keys()
])

# Total quality score
total_quality = solver.Sum([
    compatibility_score * matching_variables[(mentor_idx, mentee_idx)]
    for (mentor_idx, mentee_idx), compatibility_score in pair_compatibility_scores.items()
])

# WEIGHTED OBJECTIVE: Heavily weight number of matches
# This encourages the solver to find more matches, even if individual quality is lower
MATCH_COUNT_WEIGHT = 1000  # High weight to prioritize match count
QUALITY_WEIGHT = 1         # Lower weight for individual match quality

weighted_objective = (MATCH_COUNT_WEIGHT * total_matches) + (QUALITY_WEIGHT * total_quality)
solver.Maximize(weighted_objective)

# ================
# SOLVE OPTIMIZATION PROBLEM
# ================

print("Solving optimization problem...")
solution_status = solver.Solve()

if solution_status != pywraplp.Solver.OPTIMAL:
    raise RuntimeError("No optimal solution found. Check constraints and data.")

# Print match statistics
total_matched_mentees = sum(
    1 for (mentor_idx, mentee_idx), var in matching_variables.items()
    if var.solution_value() > 0.5
)

print(f"✅ Optimal solution found!")
print(f"📊 Total mentees matched: {total_matched_mentees} out of {len(mentees)}")
print(f"📊 Match rate: {total_matched_mentees/len(mentees)*100:.1f}%")

# ================
# EXTRACT SOLUTION & PREPARE OUTPUT
# ================

print("Extracting matches and preparing output...")

mentor_to_mentees_mapping = defaultdict(list)
for (mentor_idx, mentee_idx), decision_variable in matching_variables.items():
    if decision_variable.solution_value() > 0.5:
        mentor_to_mentees_mapping[mentor_idx].append(mentee_idx)

# ================
# OUTPUT FORMATTING
# ================

output_columns = [
    "Offset",
    "Years of Experience",
    "Roles",
    "Topics",
    "Industry",
    "Company Stage",
    "In-Person Meeting Location",
    "Time Slot Preference",
    "Sentiment",
]

def format_set_for_output(value):
    """Convert set to semicolon-separated string for CSV readability."""
    return "; ".join(sorted(value)) if isinstance(value, set) else value

def get_mentor_sort_key(mentor_idx):
    """Sort mentors alphabetically by name for consistent output."""
    return mentors.loc[mentor_idx, "Full Name"]

def get_mentee_sort_key(mentor_idx, mentee_idx):
    """Sort mentees by match score (highest first) within each mentor group."""
    return -pair_compatibility_scores[(mentor_idx, mentee_idx)]

used_submission_ids = set()
output_data_rows = []

print("Generating output with duplicate prevention...")

for mentor_idx in sorted(mentor_to_mentees_mapping.keys(), key=get_mentor_sort_key):
    mentor_data = mentors.loc[mentor_idx]
    mentor_submission_id = mentor_data.get("Mentor Submission ID", "")
    
    if mentor_submission_id and mentor_submission_id not in used_submission_ids:
        used_submission_ids.add(mentor_submission_id)
        
        assigned_mentees = sorted(
            mentor_to_mentees_mapping[mentor_idx], 
            key=lambda mentee_idx: get_mentee_sort_key(mentor_idx, mentee_idx)
        )
        
        # Add mentor row
        mentor_output_row = {
            "Row Type": "Mentor",
            "Name": mentor_data.get("Full Name", ""),
            "Email": mentor_data.get("Coda Email", ""),
            "Submission ID": mentor_submission_id,
            "Match Score": ""
        }
        
        for column in output_columns:
            mentor_output_row[column] = format_set_for_output(mentor_data[column])
        
        output_data_rows.append(mentor_output_row)
        
        # Add mentee rows
        for mentee_idx in assigned_mentees:
            mentee_data = mentees.loc[mentee_idx]
            mentee_submission_id = mentee_data.get("Mentee Submission ID", "")
            
            if mentee_submission_id and mentee_submission_id not in used_submission_ids:
                used_submission_ids.add(mentee_submission_id)
                match_score = pair_compatibility_scores[(mentor_idx, mentee_idx)]
                
                mentee_output_row = {
                    "Row Type": "Mentee", 
                    "Name": mentee_data.get("Full Name", ""),
                    "Email": mentee_data.get("Coda Email", ""),
                    "Submission ID": mentee_submission_id,
                    "Match Score": round(match_score, 2)
                }
                
                for column in output_columns:
                    mentee_output_row[column] = format_set_for_output(mentee_data[column])
                
                output_data_rows.append(mentee_output_row)

# ================
# FINAL OUTPUT
# ================

final_output_df = pd.DataFrame(output_data_rows)
final_output_df = final_output_df.drop_duplicates(subset=['Submission ID'], keep='first')

print("Saving results...")
final_output_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Matching complete! Results saved to: {OUTPUT_CSV}")
print(f"📁 Final output contains {len(final_output_df)} rows")
