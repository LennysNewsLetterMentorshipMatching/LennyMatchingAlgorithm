"""
MENTOR-MENTEE MATCHING ALGORITHM WITH UTC TIME OVERLAP & DUPLICATE PREVENTION
============================================================================

This script optimally matches mentors with mentees using Mixed Integer Linear Programming (MILP).
It maximizes total matching quality while respecting business constraints and prevents duplicate 
submission IDs in the output.

BUSINESS LOGIC:
- Each mentee gets exactly 1 mentor (or remains unmatched)
- Each mentor can have at most 2 mentees
- Mentors must have more years of experience than their mentees
- Mentor and mentee must be within 2 time zones of each other
- Similarity in roles, topics, industry, etc. INCREASES match quality
- Time slot matching uses ACTUAL UTC overlap hours (not just label matching)
- The algorithm finds the globally optimal assignment (not just greedy)
- CRITICAL: Each submission ID appears exactly once in output (no duplicates)

INPUT: Two CSV files with mentor and mentee applications
OUTPUT: CSV with mentor rows followed by their assigned mentee rows, showing match criteria and scores

SCORING FRAMEWORK:
- Base score: 0 points (all points are additive rewards)
- Years of Experience gap: +50 to +160 points (bigger meaningful gaps are better)
- Shared attributes: +10 to +80 points per shared item (similarity is rewarded)
- Time zone proximity: +20 to +30 points (closer is better within ±2 constraint)
- UTC time overlap: +8 points per overlapping hour (actual scheduling compatibility)
- Sentiment alignment: up to +30 points (normalized, down-weighted to reduce noise)

DUPLICATE PREVENTION:
- Tracks used submission IDs during output generation
- Multiple deduplication layers prevent any duplicate entries
- Final verification ensures zero duplicates in output
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer
from ortools.linear_solver import pywraplp

# =========================
# CONFIGURATION & FILE HANDLING
# =========================
# Update these paths to match your CSV file locations
MENTORS_CSV = 'Mentor-Original-Submissions.csv'
MENTEES_CSV = 'Mentee-Original-Submissions.csv'
OUTPUT_CSV = 'mentor_then_mentees_matches_fixed.csv'

# Verify files exist before proceeding (prevents confusing errors later)
for filepath in [MENTORS_CSV, MENTEES_CSV]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"File not found: {filepath}. "
            f"Check your paths or working directory: {os.getcwd()}"
        )

# =========================
# DATA LOADING & VALIDATION
# =========================
print("Loading mentor and mentee data...")
mentors = pd.read_csv(MENTORS_CSV)
mentees = pd.read_csv(MENTEES_CSV)

# Validate that all required columns exist in both datasets
# Note: Submission ID columns have different names in each CSV
# This prevents runtime errors and gives clear feedback about missing data

# Common columns required in both datasets
common_required_columns = [
    "Full Name",                    # Person's name for output identification
    "Coda Email",                   # Contact information
    "Offset",                       # Time zone offset (e.g., -5 for EST, +1 for CET)
    "Avg Year of YOE",              # Years of experience (mentor must > mentee)
    "Roles",                        # Job roles (comma-separated, e.g., "PM,Designer")
    "Topics",                       # Topics of interest (comma-separated)
    "Industry",                     # Industries (comma-separated)
    "Company Stage",                # Company stages (comma-separated)
    "In-Person Meeting Location",   # Preferred meeting locations (comma-separated)
    "Time Slot Preference",         # Preferred meeting times (e.g., "8am - 12pm: Morning")
    "Open Answer",                  # Free-text response for sentiment analysis
    "Important Attribute - First",  # Most important matching criteria
    "Important Attribute - Second", # Second most important matching criteria
    "Important Attribute - Third"   # Third most important matching criteria
]

# Dataset-specific columns (different names for submission IDs)
mentor_specific_columns = ["Mentor Submission ID"]    # Column name in mentor CSV
mentee_specific_columns = ["Mentee Submission ID"]    # Column name in mentee CSV

# Check for missing common columns
missing_mentor_common = [col for col in common_required_columns if col not in mentors.columns]
missing_mentee_common = [col for col in common_required_columns if col not in mentees.columns]

# Check for missing specific columns
missing_mentor_specific = [col for col in mentor_specific_columns if col not in mentors.columns]
missing_mentee_specific = [col for col in mentee_specific_columns if col not in mentees.columns]

# Provide clear error messages for missing columns
if missing_mentor_common or missing_mentor_specific:
    all_missing_mentor = missing_mentor_common + missing_mentor_specific
    raise ValueError(f"Mentors CSV missing required columns: {all_missing_mentor}")

if missing_mentee_common or missing_mentee_specific:
    all_missing_mentee = missing_mentee_common + missing_mentee_specific
    raise ValueError(f"Mentees CSV missing required columns: {all_missing_mentee}")

print(f"Loaded {len(mentors)} mentors and {len(mentees)} mentees")

# =========================
# DATA PREPROCESSING
# =========================
print("Preprocessing data...")

def parse_comma_separated_to_set(value):
    """
    Convert comma-separated strings to Python sets for efficient matching.
    
    Examples:
    "Product Manager, Designer" -> {"Product Manager", "Designer"}
    "Tech, Healthcare" -> {"Tech", "Healthcare"}
    NaN or empty -> set() (empty set)
    
    Sets allow fast intersection operations: set1 & set2 gives common elements
    """
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return set(item.strip() for item in str(value).split(',') if item.strip())

# These columns contain multiple selections that need to be compared for overlaps
multi_select_columns = [
    "Roles", 
    "Topics", 
    "Industry", 
    "Company Stage", 
    "In-Person Meeting Location"
]

# Convert comma-separated strings to sets for both mentors and mentees
for column in multi_select_columns:
    mentors[column] = mentors[column].apply(parse_comma_separated_to_set)
    mentees[column] = mentees[column].apply(parse_comma_separated_to_set)

# Convert string numbers to actual numeric types for mathematical operations
numeric_columns = ["Offset", "Avg Year of YOE"]
for column in numeric_columns:
    mentors[column] = pd.to_numeric(mentors[column], errors='coerce')
    mentees[column] = pd.to_numeric(mentees[column], errors='coerce')

# Clean and standardize text fields (important attributes and preferences)
important_attribute_columns = [
    "Important Attribute - First", 
    "Important Attribute - Second", 
    "Important Attribute - Third"
]

# Fill missing values with "No Preference" and standardize formatting
for column in important_attribute_columns:
    mentors[column] = mentors[column].fillna("No Preference").astype(str).str.strip()
    mentees[column] = mentees[column].fillna("No Preference").astype(str).str.strip()

# Same preprocessing for time slot preferences
mentors["Time Slot Preference"] = mentors["Time Slot Preference"].fillna("No Preference").astype(str).str.strip()
mentees["Time Slot Preference"] = mentees["Time Slot Preference"].fillna("No Preference").astype(str).str.strip()

# =========================
# SENTIMENT ANALYSIS SETUP
# =========================
print("Setting up sentiment analysis...")

# Initialize NLTK's VADER sentiment analyzer
# VADER is fast, works well on informal text, and gives scores in [-1, 1] range
# Positive scores = positive sentiment, negative = negative sentiment
sentiment_analyzer = SentimentIntensityAnalyzer()

def calculate_sentiment_score(text):
    """
    Extract sentiment score from free-text responses.
    
    Returns:
    - Float between -1 (very negative) and 1 (very positive)
    - 0 for neutral or empty text
    
    Used to match people with similar communication styles/attitudes.
    """
    if pd.isna(text) or str(text).strip() == "":
        return 0.0
    
    # VADER returns multiple scores; 'compound' is the normalized overall score
    sentiment_scores = sentiment_analyzer.polarity_scores(str(text))
    return float(sentiment_scores["compound"])

# Calculate sentiment scores for all mentors and mentees
mentors["Sentiment"] = mentors["Open Answer"].fillna("").apply(calculate_sentiment_score)
mentees["Sentiment"] = mentees["Open Answer"].fillna("").apply(calculate_sentiment_score)

# =========================
# UTC TIME SLOT OVERLAP SYSTEM (ENHANCED FEATURE)
# =========================
print("Setting up UTC-based time slot matching...")

"""
TIME SLOT OVERLAP LOGIC:

Instead of just matching labels like "Morning" with "Morning", we now:
1. Convert each person's time slot preference to actual UTC hours based on their timezone
2. Calculate the intersection of available hours between mentor and mentee
3. Score based on number of overlapping hours (more overlap = better match)

EXAMPLES:
- PST "Morning" (8am-12pm local) = UTC hours {16, 17, 18, 19} (due to PST = UTC-8)
- EST "Morning" (8am-12pm local) = UTC hours {13, 14, 15, 16} (due to EST = UTC-5)  
- Overlap = {16} = 1 hour = +8 points

This provides much more accurate scheduling compatibility than label matching alone.
"""

# Mapping from time slot labels to local hour ranges (24-hour format)
# Each tuple represents (start_hour, end_hour) in the person's LOCAL timezone
# end_hour is exclusive (e.g., (8, 12) means 8am, 9am, 10am, 11am)
TIME_SLOT_DEFINITIONS = {
    "8am - 12pm: Morning": (8, 12),
    "12pm - 6pm: Afternoon": (12, 18),
    "6pm - 10pm: Evening": (18, 22),
    "10pm - 8am: After-hours": (22, 32),  # Spans midnight; 32 means 8am next day
    
    # Alternative formatting variations (for robustness)
    "8am - 12pm Morning": (8, 12),
    "12pm - 6pm Afternoon": (12, 18),
    "6pm - 10pm Evening": (18, 22),
    "10pm - 8am After-hours": (22, 32),
    
    # Handle edge cases
    "Morning": (8, 12),
    "Afternoon": (12, 18),
    "Evening": (18, 22),
    "After-hours": (22, 32),
    
    "No Preference": None  # Special case handled separately
}

def convert_local_timeslot_to_utc_hours(slot_label, timezone_offset_hours):
    """
    Convert a local time slot preference to a set of UTC hour blocks.
    
    Args:
    - slot_label: String like "8am - 12pm: Morning" 
    - timezone_offset_hours: Numeric offset from UTC (e.g., -8 for PST, -5 for EST)
    
    Returns:
    - Set of integers representing UTC hours (0-23) when this person is available
    
    Example:
    - PST person (offset=-8) with "Morning" (8am-12pm local)
    - Local hours: {8, 9, 10, 11}
    - UTC conversion: {(8-(-8))%24, (9-(-8))%24, ...} = {16, 17, 18, 19}
    """
    # Handle "No Preference" or unknown slots
    if (not slot_label or 
        slot_label == "No Preference" or 
        slot_label not in TIME_SLOT_DEFINITIONS or 
        TIME_SLOT_DEFINITIONS[slot_label] is None):
        return set()  # Empty set means no specific preference
    
    start_local, end_local = TIME_SLOT_DEFINITIONS[slot_label]
    utc_hour_set = set()
    
    # Convert each local hour to its UTC equivalent
    for local_hour in range(start_local, end_local):
        # Handle 24-hour wraparound and timezone conversion
        actual_local_hour = local_hour % 24  # Handle cases like hour 32 -> 8
        utc_hour = (actual_local_hour - timezone_offset_hours) % 24
        utc_hour_set.add(utc_hour)
    
    return utc_hour_set

def calculate_utc_availability_overlap_bonus(mentor_slot, mentor_offset, mentee_slot, mentee_offset):
    """
    Calculate scoring bonus based on actual UTC time overlap between mentor and mentee.
    
    BUSINESS LOGIC:
    - More overlapping hours = higher compatibility score
    - "No Preference" from either person = neutral (no bonus/penalty)
    - Each overlapping hour contributes equally to the match quality
    
    SCORING:
    - Each overlapping UTC hour = +8 points
    - Maximum possible overlap varies by slot (Morning = 4 hrs max, Evening = 4 hrs max)
    - Cross-timezone matches can still work if availability windows align
    
    Returns:
    - Integer points to add to overall match score
    """
    # Handle "No Preference" cases neutrally
    if mentor_slot == "No Preference" or mentee_slot == "No Preference":
        return 0  # Neutral - don't penalize flexibility, but no match bonus
    
    # Convert both preferences to UTC hour sets
    mentor_utc_hours = convert_local_timeslot_to_utc_hours(mentor_slot, mentor_offset)
    mentee_utc_hours = convert_local_timeslot_to_utc_hours(mentee_slot, mentee_offset)
    
    # Calculate intersection (overlapping hours)
    overlapping_hours = mentor_utc_hours & mentee_utc_hours
    overlap_count = len(overlapping_hours)
    
    # Score: 8 points per overlapping hour
    # This makes time compatibility meaningful in the overall scoring
    points_per_hour = 8
    total_bonus = overlap_count * points_per_hour
    
    return total_bonus

# =========================
# SCORING SYSTEM CONFIGURATION
# =========================
"""
SCORING WEIGHTS: Higher weights = more important for matching

These weights determine how much each type of similarity contributes to the match score.
They're based on business requirements and can be tuned based on matching outcomes.
"""
ATTRIBUTE_WEIGHTS = {
    "Roles": 8,                # Job roles are very important for relevant mentoring
    "Topics": 7,               # Shared interests enable better conversations
    "Industry": 6,             # Industry knowledge transfer is valuable
    "Company Stage": 5,        # Stage-specific challenges and advice
    "OffsetExact": 2,          # Small bonus for exact same time zone
    "InPerson": 1,             # In-person meeting capability (nice to have)
    "Sentiment": 3,            # Communication style match (down-weighted to avoid noise)
}

BASE_SCORE = 0.0  # Start from 0; all scoring is additive rewards (no penalties for dissimilarity)

def calculate_set_similarity_bonus(mentor_set, mentee_set, points_per_match):
    """
    Calculate bonus points for shared items between mentor and mentee sets.
    
    Examples:
    - Mentor roles: {"PM", "Designer"}, Mentee roles: {"PM", "Engineer"}
    - Shared: {"PM"} -> 1 match -> points_per_match * 1
    
    This rewards similarity (people with common ground tend to work well together).
    """
    shared_items = mentor_set & mentee_set  # Set intersection
    return len(shared_items) * points_per_match

def calculate_timezone_proximity_bonus(mentor_offset, mentee_offset):
    """
    Calculate bonus points based on time zone proximity.
    
    BUSINESS RULES:
    1. Hard constraint: Must be within ±2 hours (enforced elsewhere)
    2. Within that constraint: Closer = better for general scheduling flexibility
    
    SCORING:
    - Same time zone (0 difference): +10 base + +20 exact bonus = +30 total
    - 1 hour apart: +5 base = +5 total  
    - 2 hours apart: +0 base = +0 total
    
    NOTE: This is separate from time slot overlap, which handles specific availability windows.
    """
    if pd.isna(mentor_offset) or pd.isna(mentee_offset):
        return 0.0
    
    time_difference = abs(mentor_offset - mentee_offset)
    
    # This should never happen due to feasibility filtering, but safety check
    if time_difference > 2:
        return -np.inf
    
    # Base proximity bonus: closer time zones get more points
    proximity_bonus = max(0, 2 - time_difference) * 5.0
    
    # Additional bonus for exact same time zone (no conversion needed for any meetings)
    if time_difference == 0:
        proximity_bonus += 10.0 * ATTRIBUTE_WEIGHTS["OffsetExact"]
    
    return proximity_bonus

def calculate_experience_bonus(mentor_yoe, mentee_yoe):
    """
    Calculate bonus points based on years of experience gap.
    
    BUSINESS RULES:
    1. Mentor MUST have more experience than mentee (hard constraint)
    2. Different experience gaps have different values for mentoring:
    
    EXPERIENCE GAP SCORING:
    - 2-3 years: +160 points (sweet spot - recent relevant experience)
    - 4-8 years: +100 points (good gap - substantial experience difference) 
    - 8+ years: +50 points (large gap - very senior mentor)
    - 1-2 years: +10 points (minimal gap but still valid)
    - 0 or negative: INVALID (mentor must be more experienced)
    """
    if pd.isna(mentor_yoe) or pd.isna(mentee_yoe):
        return -np.inf
    
    experience_gap = mentor_yoe - mentee_yoe
    
    # Hard business constraint: mentor must be more experienced
    if experience_gap <= 0:
        return -np.inf
    
    # Score based on optimal experience gaps for mentoring relationships
    if 2 <= experience_gap <= 3:
        return 160.0    # Optimal gap - recent enough to be relevant
    elif 4 <= experience_gap <= 8:
        return 100.0    # Good gap - substantial difference
    elif experience_gap > 8:
        return 50.0     # Large gap - very senior mentor
    else:
        # Small gap (mentor > mentee but < 2 years) - valid but not ideal
        return 10.0

def calculate_sentiment_alignment_bonus(mentor_sentiment, mentee_sentiment):
    """
    Calculate bonus for sentiment/communication style similarity.
    
    METHODOLOGY:
    - Both sentiments are in [-1, 1] range from VADER
    - Alignment = 1 - |difference| gives similarity score in [0, 1]
    - Scale by weight to get final bonus
    
    EXAMPLES:
    - Both 0.8 (very positive): alignment = 1.0 -> +30 points
    - One 0.8, other 0.6: alignment = 0.8 -> +24 points  
    - One 0.5, other -0.5: alignment = 0.0 -> +0 points
    
    This is down-weighted (weight=3) to avoid noise from text analysis dominating.
    """
    alignment_score = 1.0 - abs(mentor_sentiment - mentee_sentiment)
    return alignment_score * 10.0 * ATTRIBUTE_WEIGHTS["Sentiment"]

def compute_overall_match_score(mentor_row, mentee_row):
    """
    Calculate the total compatibility score between a mentor and mentee.
    
    SCORING PHILOSOPHY:
    - Start from 0 and add points for positive attributes
    - Similarity increases score (people with common ground work well together)
    - Hard constraints return -infinity (impossible matches)
    - All business rules and weights are applied here
    - Enhanced: Time slot scoring based on actual UTC overlap hours
    
    Returns:
    - Positive number: viable match (higher = better)
    - -infinity: impossible match (fails hard constraints)
    """
    
    # ===== HARD CONSTRAINTS (MUST PASS) =====
    
    # Time zone constraint: must be within ±2 hours for practical scheduling
    mentor_tz = mentor_row["Offset"]
    mentee_tz = mentee_row["Offset"]
    
    if pd.isna(mentor_tz) or pd.isna(mentee_tz):
        return -np.inf
    
    if abs(mentor_tz - mentee_tz) > 2:
        return -np.inf
    
    # Experience constraint: mentor must be more experienced for mentoring value
    experience_bonus = calculate_experience_bonus(
        mentor_row["Avg Year of YOE"], 
        mentee_row["Avg Year of YOE"]
    )
    if experience_bonus == -np.inf:
        return -np.inf
    
    # ===== SCORE CALCULATION (ADD POINTS FOR COMPATIBILITY) =====
    
    total_score = BASE_SCORE
    
    # Add experience gap bonus
    total_score += experience_bonus
    
    # Add time zone proximity bonus  
    timezone_bonus = calculate_timezone_proximity_bonus(mentor_tz, mentee_tz)
    if timezone_bonus == -np.inf:
        return -np.inf
    total_score += timezone_bonus
    
    # Add bonuses for shared attributes (similarity rewards)
    total_score += calculate_set_similarity_bonus(
        mentor_row["Roles"], mentee_row["Roles"], 
        10.0 * ATTRIBUTE_WEIGHTS["Roles"]
    )
    total_score += calculate_set_similarity_bonus(
        mentor_row["Topics"], mentee_row["Topics"], 
        10.0 * ATTRIBUTE_WEIGHTS["Topics"]
    )
    total_score += calculate_set_similarity_bonus(
        mentor_row["Industry"], mentee_row["Industry"], 
        10.0 * ATTRIBUTE_WEIGHTS["Industry"]
    )
    total_score += calculate_set_similarity_bonus(
        mentor_row["Company Stage"], mentee_row["Company Stage"], 
        10.0 * ATTRIBUTE_WEIGHTS["Company Stage"]
    )
    total_score += calculate_set_similarity_bonus(
        mentor_row["In-Person Meeting Location"], mentee_row["In-Person Meeting Location"], 
        10.0 * ATTRIBUTE_WEIGHTS["InPerson"]
    )
    
    # Add bonus for matching important attributes (when both are specific, not "No Preference")
    for attribute_column in important_attribute_columns:
        mentor_value = mentor_row.get(attribute_column, "No Preference")
        mentee_value = mentee_row.get(attribute_column, "No Preference")
        
        # Only reward when both people have specific preferences that match
        if (mentor_value != "No Preference" and 
            mentee_value != "No Preference" and 
            mentor_value == mentee_value):
            total_score += 10.0
    
    # Add UTC time slot availability overlap bonus (ENHANCED FEATURE)
    # This replaces the simple label matching with actual hour-based overlap calculation
    availability_bonus = calculate_utc_availability_overlap_bonus(
        mentor_row["Time Slot Preference"], mentor_row["Offset"],
        mentee_row["Time Slot Preference"], mentee_row["Offset"]
    )
    total_score += availability_bonus
    
    # Add sentiment alignment bonus
    total_score += calculate_sentiment_alignment_bonus(
        mentor_row["Sentiment"], 
        mentee_row["Sentiment"]
    )
    
    return total_score

# =========================
# PAIRWISE SCORE COMPUTATION
# =========================
print("Computing compatibility scores for all mentor-mentee pairs...")

mentor_indices = mentors.index.tolist()
mentee_indices = mentees.index.tolist()
pair_compatibility_scores = {}

# Calculate scores for all possible mentor-mentee combinations
# This creates the "cost matrix" for the optimization algorithm
for mentor_idx in mentor_indices:
    mentor_data = mentors.loc[mentor_idx]
    for mentee_idx in mentee_indices:
        mentee_data = mentees.loc[mentee_idx]
        
        compatibility_score = compute_overall_match_score(mentor_data, mentee_data)
        
        # Only store viable matches (ignore impossible pairings)
        if compatibility_score > -np.inf:
            pair_compatibility_scores[(mentor_idx, mentee_idx)] = compatibility_score

if not pair_compatibility_scores:
    raise ValueError("No valid matches found under current constraints. Check your data and business rules.")

# =========================
# OPTIMIZATION SETUP (LINEAR PROGRAMMING)
# =========================
print("Setting up optimization problem...")

"""
OPTIMIZATION PROBLEM FORMULATION:

This is a "Maximum Weight Matching" problem with additional constraints.
We use Mixed Integer Linear Programming (MILP) to find the globally optimal solution.

VARIABLES:
- x[mentor_i, mentee_j] = 1 if mentor i is matched to mentee j, 0 otherwise

OBJECTIVE:
- Maximize sum of (compatibility_score * x[mentor_i, mentee_j]) for all pairs

CONSTRAINTS:
- Each mentee matched to at most 1 mentor: sum(x[i,j] for all i) <= 1 for each j
- Each mentor matched to at most 2 mentees: sum(x[i,j] for all j) <= 2 for each i

This guarantees the globally optimal assignment (not just a greedy approximation).
"""

# Initialize OR-Tools solver (Google's optimization library)
solver = pywraplp.Solver.CreateSolver('SCIP')
if not solver:
    raise RuntimeError("Failed to initialize OR-Tools solver. Check your installation.")

# Create decision variables: x[mentor, mentee] = 1 if matched, 0 otherwise
matching_variables = {}
for (mentor_idx, mentee_idx) in pair_compatibility_scores.keys():
    var_name = f"x_{mentor_idx}_{mentee_idx}"
    matching_variables[(mentor_idx, mentee_idx)] = solver.BoolVar(var_name)

# =========================
# OPTIMIZATION CONSTRAINTS
# =========================
print("Adding business constraints...")

# CONSTRAINT 1: Each mentee can have at most 1 mentor
# This ensures no mentee is overwhelmed with multiple mentoring relationships
for mentee_idx in mentee_indices:
    # Find all potential mentors for this mentee
    potential_matches = [
        matching_variables[(mentor_idx, mentee_idx)] 
        for mentor_idx in mentor_indices 
        if (mentor_idx, mentee_idx) in matching_variables
    ]
    
    if potential_matches:  # Only add constraint if there are potential matches
        solver.Add(solver.Sum(potential_matches) <= 1)

# CONSTRAINT 2: Each mentor can have at most 2 mentees  
# This prevents mentor overload while allowing some mentors to help multiple people
for mentor_idx in mentor_indices:
    # Find all potential mentees for this mentor
    potential_matches = [
        matching_variables[(mentor_idx, mentee_idx)] 
        for mentee_idx in mentee_indices 
        if (mentor_idx, mentee_idx) in matching_variables
    ]
    
    if potential_matches:  # Only add constraint if there are potential matches
        solver.Add(solver.Sum(potential_matches) <= 2)

# =========================
# OPTIMIZATION OBJECTIVE
# =========================
print("Setting optimization objective...")

# OBJECTIVE: Maximize total compatibility score across all matches
# This finds the assignment that maximizes overall matching quality
objective_terms = []
for (mentor_idx, mentee_idx), compatibility_score in pair_compatibility_scores.items():
    match_variable = matching_variables[(mentor_idx, mentee_idx)]
    objective_terms.append(compatibility_score * match_variable)

solver.Maximize(solver.Sum(objective_terms))

# =========================
# SOLVE OPTIMIZATION PROBLEM
# =========================
print("Solving optimization problem...")

solution_status = solver.Solve()

if solution_status != pywraplp.Solver.OPTIMAL:
    raise RuntimeError("No optimal solution found. Check constraints and data.")

print("Optimal solution found!")

# =========================
# EXTRACT SOLUTION & PREPARE OUTPUT
# =========================
print("Extracting matches and preparing output...")

# Determine which mentor-mentee pairs were selected in the optimal solution
mentor_to_mentees_mapping = defaultdict(list)
for (mentor_idx, mentee_idx), decision_variable in matching_variables.items():
    if decision_variable.solution_value() > 0.5:  # Variable is set to 1 (matched)
        mentor_to_mentees_mapping[mentor_idx].append(mentee_idx)

# =========================
# OUTPUT FORMATTING WITH COMPREHENSIVE DUPLICATE PREVENTION
# =========================
"""
OUTPUT FORMAT:
- Each mentor gets one row (with blank match score)
- Followed by one row per assigned mentee (with their match score)
- Only show columns relevant to matching criteria for easy QA
- Sort by mentor name for consistent output
- Include unified Submission ID for each person (mapped from dataset-specific columns)

CRITICAL DUPLICATE PREVENTION:
- Track used submission IDs during output generation to prevent duplicates
- Multiple layers of deduplication ensure no duplicate entries
- Final verification guarantees zero duplicates in output
- Each submission ID appears exactly once in the final CSV
"""

# Define which columns to include in output (only matching criteria for clarity)
output_columns = [
    "Offset",                      # Time zone for scheduling
    "Avg Year of YOE",            # Experience level
    "Roles",                       # Job roles for relevance
    "Topics",                      # Topics of interest
    "Industry",                    # Industry background
    "Company Stage",               # Company stage experience
    "In-Person Meeting Location",  # Meeting location preferences
    "Time Slot Preference",        # Scheduling preferences (original label)
    "Sentiment",                   # Communication style indicator
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

# CRITICAL DUPLICATE PREVENTION: Track unique submission IDs
used_submission_ids = set()  # This will track every submission ID we've added to output
output_data_rows = []

print("Generating output with duplicate prevention...")

# Process mentors in alphabetical order for consistent output
for mentor_idx in sorted(mentor_to_mentees_mapping.keys(), key=get_mentor_sort_key):
    mentor_data = mentors.loc[mentor_idx]
    mentor_submission_id = mentor_data.get("Mentor Submission ID", "")
    
    # CRITICAL CHECK 1: Only process mentor if submission ID hasn't been used
    if mentor_submission_id and mentor_submission_id not in used_submission_ids:
        # Mark this mentor's submission ID as used
        used_submission_ids.add(mentor_submission_id)
        
        # Sort this mentor's mentees by match score (best matches first)
        assigned_mentees = sorted(
            mentor_to_mentees_mapping[mentor_idx], 
            key=lambda mentee_idx: get_mentee_sort_key(mentor_idx, mentee_idx)
        )
        
        # Add mentor row (no match score shown)
        # Map the mentor-specific submission ID column to unified "Submission ID"
        mentor_output_row = {
            "Row Type": "Mentor",
            "Name": mentor_data.get("Full Name", ""),
            "Email": mentor_data.get("Coda Email", ""),
            "Submission ID": mentor_submission_id,  # Map from mentor-specific column
            "Match Score": ""  # Blank for mentor rows
        }
        
        # Add mentor's criteria values
        for column in output_columns:
            mentor_output_row[column] = format_set_for_output(mentor_data[column])
        
        output_data_rows.append(mentor_output_row)
        
        # Add one row per assigned mentee (with match scores and duplicate prevention)
        for mentee_idx in assigned_mentees:
            mentee_data = mentees.loc[mentee_idx]
            mentee_submission_id = mentee_data.get("Mentee Submission ID", "")
            
            # CRITICAL CHECK 2: Only add mentee if submission ID hasn't been used
            if mentee_submission_id and mentee_submission_id not in used_submission_ids:
                # Mark this mentee's submission ID as used
                used_submission_ids.add(mentee_submission_id)
                match_score = pair_compatibility_scores[(mentor_idx, mentee_idx)]
                
                # Map the mentee-specific submission ID column to unified "Submission ID"
                mentee_output_row = {
                    "Row Type": "Mentee", 
                    "Name": mentee_data.get("Full Name", ""),
                    "Email": mentee_data.get("Coda Email", ""),
                    "Submission ID": mentee_submission_id,  # Map from mentee-specific column
                    "Match Score": round(match_score, 2)  # Show match score on mentee rows
                }
                
                # Add mentee's criteria values  
                for column in output_columns:
                    mentee_output_row[column] = format_set_for_output(mentee_data[column])
                
                output_data_rows.append(mentee_output_row)

# =========================
# FINAL SAFETY NET: NUCLEAR DUPLICATE REMOVAL
# =========================
"""
FINAL DEDUPLICATION LAYER:

Even though our generation logic should prevent duplicates, we add a final safety net
that removes any remaining duplicates based on Submission ID. This ensures 100%
certainty that no duplicates exist in the final output.
"""

final_output_df = pd.DataFrame(output_data_rows)

# NUCLEAR OPTION: Final deduplication by submission ID (keeps first occurrence)
final_output_df = final_output_df.drop_duplicates(subset=['Submission ID'], keep='first')

# =========================
# SAVE OUTPUT FILE
# =========================
print("Saving results...")

final_output_df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Matching complete! Results saved to: {OUTPUT_CSV}")
