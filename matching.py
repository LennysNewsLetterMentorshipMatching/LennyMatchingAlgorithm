"""
MENTOR-MENTEE MATCHING ALGORITHM - COMPLETE VERSION WITH INDIVIDUAL OPEN ANSWERS
====
This version includes each person's individual "Open Answer" text in their respective row
for efficient manual quality assurance and match verification.

Key Features:
- Text-based goal matching (no PyTorch dependencies)
- New mentee prioritization (+50 points bonus)
- Soft timezone constraints (graduated penalties)
- Company stage matching excludes "No Preference"
- Individual Open Answers column for QA (each person shows their own text)
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

# Define input and output file paths
MENTORS_CSV = 'Mentor-Original-Submissions.csv'
MENTEES_CSV = 'Mentee-Original-Submissions.csv'
OUTPUT_CSV = 'mentor_then_mentees_matches_fixed.csv'

# Verify that required input files exist before proceeding
for filepath in [MENTORS_CSV, MENTEES_CSV]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

print("✅ Input files found and verified")

# ================
# DATA LOADING & INITIAL VALIDATION
# ================

print("Loading mentor and mentee data...")
mentors = pd.read_csv(MENTORS_CSV)
mentees = pd.read_csv(MENTEES_CSV)

# Handle timezone column naming differences between CSV files
# Mentors CSV uses "Timezones", Mentees CSV uses "Timezone"
if "Timezones" in mentors.columns:
    mentors["Timezone"] = mentors["Timezones"]  # Standardize to "Timezone"
    print("✅ Mapped 'Timezones' to 'Timezone' for mentors")
elif "Timezone" not in mentors.columns:
    raise ValueError("Neither 'Timezone' nor 'Timezones' column found in mentors CSV")

if "Timezone" not in mentees.columns:
    raise ValueError("'Timezone' column not found in mentees CSV")

print(f"✅ Loaded {len(mentors)} mentors and {len(mentees)} mentees")

# ================
# TIMEZONE MAPPING & PARSING FUNCTIONS
# ================

# Map common timezone strings to UTC offset values for numerical comparison
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
    """
    Convert timezone strings to numeric UTC offsets for mathematical comparison.
    
    Args:
        timezone_str: String like "EST - Eastern US/Canada" or numeric value
        
    Returns:
        float: UTC offset (e.g., -5.0 for EST) or np.nan if unparseable
    """
    if pd.isna(timezone_str):
        return np.nan
    
    timezone_str = str(timezone_str).strip()
    
    # Try direct mapping from our timezone dictionary
    if timezone_str in TIMEZONE_OFFSETS:
        return TIMEZONE_OFFSETS[timezone_str]
    
    # Try to parse as numeric value (already an offset)
    try:
        return float(timezone_str)
    except:
        return 0  # Default to UTC if unparseable

# ================
# DATA PARSING & PREPROCESSING FUNCTIONS
# ================

def parse_comma_separated_to_set(value):
    """
    Convert comma-separated strings to Python sets for efficient matching.
    
    Examples:
    - "Product Manager, Designer" → {"Product Manager", "Designer"}  
    - "Tech, Healthcare" → {"Tech", "Healthcare"}
    - NaN or empty → set() (empty set)
    
    Sets allow fast intersection operations: set1 & set2 gives common elements
    """
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return set(item.strip() for item in str(value).split(',') if item.strip())

def parse_years_of_experience(value):
    """
    Parse years of experience from various formats into numeric values.
    
    Handles:
    - Ranges: "3-5" → 4.0 (average)
    - Plus signs: "20+" → 20.0  
    - Direct numbers: "5" → 5.0
    - Invalid/missing → np.nan
    """
    if pd.isna(value):
        return np.nan
    
    value_str = str(value).strip()
    
    # Handle ranges like "3-5", "6-8"
    if '-' in value_str:
        try:
            parts = value_str.split('-')
            if len(parts) == 2:
                return (float(parts[0]) + float(parts[1])) / 2
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

def parse_previous_matches(value):
    """
    Determine if a mentee has been matched before based on "Previous Matches" field.
    
    Returns:
    - True: Has previous matches ("Yes - as a mentee", etc.)
    - False: No previous matches ("No", "Yes - as a mentor", etc.) → Gets priority bonus
    """
    if pd.isna(value):
        return False  # Assume new mentee if missing data
    
    value_str = str(value).strip().lower()
    
    # Indicators of previous mentoring experience (as mentee)
    has_previous_indicators = ['yes', 'yes - as a mentee', 'previously matched']
    
    # Indicators of no previous mentoring (or only as mentor) → prioritize these
    no_previous_indicators = ['no', 'yes - as a mentor', 'first time']
    
    for indicator in has_previous_indicators:
        if indicator in value_str:
            return True  # Has previous matches
    
    return False  # Default to new mentee (gets priority)

# ================
# APPLY DATA PREPROCESSING
# ================

print("Preprocessing and cleaning data...")

# Convert timezone strings to numeric offsets
mentors["Offset"] = mentors["Timezone"].apply(parse_timezone_to_offset)
mentees["Offset"] = mentees["Timezone"].apply(parse_timezone_to_offset)

# Parse years of experience into numeric values
mentors["Years of Experience"] = mentors["Years of Experience"].apply(parse_years_of_experience)
mentees["Years of Experience"] = mentees["Years of Experience"].apply(parse_years_of_experience)

# Parse previous match status for mentees (affects prioritization)
mentees["Has_Previous_Matches"] = mentees["Previous Matches"].apply(parse_previous_matches)

# Create full names for display purposes
mentors["Full Name"] = mentors["First Name"].astype(str) + " " + mentors["Last Name"].astype(str)
mentees["Full Name"] = mentees["First Name"].astype(str) + " " + mentees["Last Name"].astype(str)

# Convert multi-select fields to sets for efficient intersection operations
multi_select_columns = ["Roles", "Topics", "Industry", "Company Stage", "In-Person Meeting Location"]
for column in multi_select_columns:
    mentors[column] = mentors[column].apply(parse_comma_separated_to_set)
    mentees[column] = mentees[column].apply(parse_comma_separated_to_set)

# Clean and standardize text fields (important attributes)
important_attribute_columns = [
    "Important Attribute - First", "Important Attribute - Second", "Important Attribute - Third"
]

for column in important_attribute_columns:
    mentors[column] = mentors[column].fillna("No Preference").astype(str).str.strip()
    mentees[column] = mentees[column].fillna("No Preference").astype(str).str.strip()

# Clean time slot preferences
mentors["Time Slot Preference"] = mentors["Time Slot Preference"].fillna("No Preference").astype(str).str.strip()
mentees["Time Slot Preference"] = mentees["Time Slot Preference"].fillna("No Preference").astype(str).str.strip()

# Display mentee status breakdown for transparency
new_mentees_count = (~mentees["Has_Previous_Matches"]).sum()
returning_mentees_count = mentees["Has_Previous_Matches"].sum()
print(f"📊 Mentee Status: {new_mentees_count} new mentees, {returning_mentees_count} returning mentees")
print(f"✅ New mentees will receive +50 point priority bonus")

# ================
# TEXT-BASED GOAL ANALYSIS SYSTEM
# ================

def extract_keywords(text):
    """
    Extract meaningful keywords from text for semantic matching.
    
    Process:
    1. Convert to lowercase and extract words (3+ characters)
    2. Remove common stop words that don't indicate goals/interests
    3. Return set of meaningful terms for comparison
    
    This enables matching people with similar goals without ML dependencies.
    """
    if pd.isna(text) or str(text).strip() == "":
        return set()
    
    text = str(text).lower()
    
    # Extract words of 3+ characters using regex
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    
    # Comprehensive stop words list (words that don't indicate goals)
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
    
    # Filter out stop words and keep meaningful terms (4+ characters for better signal)
    meaningful_words = set(word for word in words if word not in stop_words and len(word) > 3)
    
    return meaningful_words

def calculate_enhanced_text_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts using keyword analysis.
    
    Algorithm:
    1. Extract meaningful keywords from both texts
    2. Calculate Jaccard similarity (intersection/union)
    3. Calculate weighted overlap (longer shared words = higher weight)
    4. Combine scores for final similarity measure
    
    Returns: Float 0.0-1.0 (higher = more similar goals/interests)
    """
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)
    
    if len(keywords1) == 0 or len(keywords2) == 0:
        return 0.0  # No meaningful content to compare
    
    # Jaccard similarity: shared_words / total_unique_words
    intersection = len(keywords1 & keywords2)
    union = len(keywords1 | keywords2)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Weighted overlap: longer shared keywords get higher weight
    shared_keywords = keywords1 & keywords2
    weight_score = sum(len(word) for word in shared_keywords) / max(
        sum(len(word) for word in keywords1), 
        sum(len(word) for word in keywords2), 
        1
    )
    
    # Combine scores (favor Jaccard but boost for meaningful long words)
    final_similarity = (jaccard * 0.7) + (weight_score * 0.3)
    return min(final_similarity, 1.0)

# ================
# GOAL TOPIC CATEGORIZATION
# ================

def simple_topic_assignment(texts):
    """
    Assign goal topic categories based on keyword analysis.
    
    Categories:
    0. General (default)
    1. Leadership (management, team leadership)  
    2. Technical (engineering, development)
    3. Product/Strategy (product management, business strategy)
    4. Startup (entrepreneurship, founding)
    5. Career Transition (career changes, switching roles)
    
    This enables bonus points for people with exactly matching goal categories.
    """
    topics = []
    
    for text in texts:
        keywords = extract_keywords(text)
        
        # Define topic detection rules based on keyword presence
        if any(word in keywords for word in ['leadership', 'manage', 'team', 'lead', 'director', 'manager']):
            topics.append(1)  # Leadership
        elif any(word in keywords for word in ['technical', 'engineering', 'developer', 'code', 'software', 'programming']):
            topics.append(2)  # Technical
        elif any(word in keywords for word in ['product', 'strategy', 'market', 'business', 'planning']):
            topics.append(3)  # Product/Strategy
        elif any(word in keywords for word in ['startup', 'entrepreneur', 'founder', 'venture', 'founding']):
            topics.append(4)  # Startup
        elif any(word in keywords for word in ['career', 'transition', 'change', 'move', 'switch', 'pivot']):
            topics.append(5)  # Career Transition
        else:
            topics.append(0)  # General/Other
    
    return topics

# Extract and analyze goal texts from Open Answer fields
print("Analyzing program goals and interests...")
mentor_goal_texts = mentors["Open Answer"].fillna("").tolist()
mentee_goal_texts = mentees["Open Answer"].fillna("").tolist()

# Assign topic categories for bonus matching
mentors["Goal_Topic"] = simple_topic_assignment(mentor_goal_texts)
mentees["Goal_Topic"] = simple_topic_assignment(mentee_goal_texts)

print("✅ Goal analysis complete - using enhanced text-based matching")

# ================
# MATCHING SCORE CALCULATION FUNCTIONS
# ================

BASE_SCORE = 0.0  # All matches start from 0, points added for compatibility

def calculate_new_mentee_bonus(has_previous_matches):
    """
    Priority bonus for mentees who haven't been matched before.
    
    Returns:
    - 50.0 points: New mentee (prioritize first-time participants)  
    - 0.0 points: Returning mentee (no bonus)
    """
    return 50.0 if not has_previous_matches else 0.0

def calculate_goal_alignment_bonus(mentor_idx, mentee_idx):
    """
    Calculate bonus points for goal/interest alignment using text similarity.
    
    Process:
    1. Get Open Answer texts for mentor and mentee
    2. Calculate enhanced text similarity (0.0-1.0)
    3. Scale to point range: similarity * 75 = 0-75 points
    
    This is the highest-weighted factor (up to 75 points).
    """
    mentor_text = mentor_goal_texts[mentor_idx]
    mentee_text = mentee_goal_texts[mentee_idx]
    similarity = calculate_enhanced_text_similarity(mentor_text, mentee_text)
    return similarity * 75  # Scale to 0-75 point range

def calculate_topic_alignment_bonus(mentor_topic, mentee_topic):
    """
    Bonus for exact topic category matches.
    
    Returns:
    - 30.0 points: Same specific topic (Leadership, Technical, etc.)
    - 0.0 points: Different topics or both "General"
    """
    if mentor_topic == mentee_topic and mentor_topic != 0:  # Don't reward generic matches
        return 30.0
    return 0.0

def calculate_experience_bonus(mentor_yoe, mentee_yoe):
    """
    Experience gap bonus - mentors must have more experience than mentees.
    
    Rules:
    1. Mentor MUST have more experience (hard constraint)
    2. Different gaps have different values:
       - 8+ years gap: 160 points (very senior mentor)
       - 4-7 years gap: 120 points (good gap) 
       - 2-3 years gap: 100 points (solid gap)
       - 1-2 years gap: 80 points (small gap)
       - <1 year gap: 50 points (minimal gap)
    
    Returns: Points or -np.inf (impossible match)
    """
    # Handle missing data with reasonable defaults
    if pd.isna(mentor_yoe) or pd.isna(mentee_yoe):
        mentor_yoe = mentor_yoe if not pd.isna(mentor_yoe) else 5.0
        mentee_yoe = mentee_yoe if not pd.isna(mentee_yoe) else 2.0
    
    experience_gap = mentor_yoe - mentee_yoe
    
    # Hard constraint: mentor must be more experienced
    if experience_gap <= 0:
        return -np.inf  # Impossible match
    
    # Score based on experience gap size
    if experience_gap >= 8: return 160.0
    elif experience_gap >= 4: return 120.0
    elif experience_gap >= 2: return 100.0
    elif experience_gap >= 1: return 80.0
    else: return 50.0

def calculate_timezone_flexibility_bonus(mentor_offset, mentee_offset):
    """
    Soft timezone constraint with graduated scoring.
    
    NEW APPROACH: No hard rejection, graduated penalties for larger differences.
    This allows great matches to overcome timezone challenges.
    
    Scoring:
    - 0-1 hours difference: +30 points (excellent)
    - 2 hours difference: +20 points (good)  
    - 3-4 hours difference: +5 points (acceptable)
    - 5-6 hours difference: -10 points (challenging)
    - 7-8 hours difference: -25 points (difficult)
    - 9+ hours difference: -40 points (very difficult but not impossible)
    """
    if pd.isna(mentor_offset) or pd.isna(mentee_offset):
        return 10.0  # Neutral bonus for missing timezone data
    
    time_difference = abs(mentor_offset - mentee_offset)
    
    # Graduated scoring (no hard cutoffs)
    if time_difference <= 1: return 30.0
    elif time_difference <= 2: return 20.0
    elif time_difference <= 4: return 5.0
    elif time_difference <= 6: return -10.0
    elif time_difference <= 8: return -25.0
    else: return -40.0

def calculate_time_slot_bonus(mentor_slot, mentee_slot):
    """
    Time slot preference matching bonus (reduced weighting).
    
    Rules:
    - Both specify same slot: +15 points (reduced from +60)
    - Either has "No Preference": 0 points (neutral)  
    - Both specify different slots: 0 points (neutral, no penalty)
    """
    if mentor_slot == "No Preference" or mentee_slot == "No Preference":
        return 0.0  # Neutral for flexible people
    
    return 15.0 if mentor_slot == mentee_slot else 0.0

def calculate_attribute_match_points(mentor_set, mentee_set, base_points=10, exclude_no_preference=False):
    """
    Calculate bonus points for shared attributes (roles, topics, industry, etc.).
    
    Args:
        mentor_set: Set of mentor's attributes
        mentee_set: Set of mentee's attributes  
        base_points: Points awarded per shared item
        exclude_no_preference: Whether to filter out "No Preference" entries
        
    Returns: Points = (number of shared items) * base_points
    """
    if exclude_no_preference:
        # Filter out "No Preference" before calculating matches (for company stage)
        mentor_filtered = {item for item in mentor_set if item.lower() != 'no preference'}
        mentee_filtered = {item for item in mentee_set if item.lower() != 'no preference'}
        shared_items = mentor_filtered & mentee_filtered
    else:
        shared_items = mentor_set & mentee_set
    
    return len(shared_items) * base_points

def compute_overall_match_score(mentor_row, mentee_row, mentor_idx, mentee_idx):
    """
    Calculate total compatibility score between a mentor and mentee.
    
    SCORING BREAKDOWN:
    - New Mentee Bonus: 0-50 points
    - Goal Alignment: 0-75 points (highest weight)
    - Topic Match: 0-30 points
    - Experience Gap: 50-160 points (or -inf for impossible)
    - Timezone Flexibility: -40 to +30 points
    - Shared Roles: 0-15 points per role
    - Shared Topics: 0-12 points per topic
    - Shared Industry: 0-10 points per industry
    - Shared Company Stage: 0-10 points per stage (excluding "No Preference")
    - Shared Locations: 0-5 points per location
    - Important Attributes: 0-15 points per exact match
    - Time Slot Match: 0-15 points
    
    Returns: Total score (higher = better match) or -np.inf (impossible)
    """
    # Check hard constraint: experience gap
    experience_bonus = calculate_experience_bonus(
        mentor_row["Years of Experience"], mentee_row["Years of Experience"]
    )
    if experience_bonus == -np.inf:
        return -np.inf  # Impossible match
    
    # Start scoring from base and add compatibility points
    total_score = BASE_SCORE
    
    # Priority bonus for new mentees (encourages inclusion)
    total_score += calculate_new_mentee_bonus(mentee_row["Has_Previous_Matches"])
    
    # Goal alignment (highest weight - up to 75 points)
    total_score += calculate_goal_alignment_bonus(mentor_idx, mentee_idx)
    
    # Topic category match bonus
    total_score += calculate_topic_alignment_bonus(mentor_row["Goal_Topic"], mentee_row["Goal_Topic"])
    
    # Experience gap bonus (major factor)
    total_score += experience_bonus
    
    # Timezone flexibility (soft constraint)
    total_score += calculate_timezone_flexibility_bonus(mentor_row["Offset"], mentee_row["Offset"])
    
    # Shared attribute bonuses (various weights based on importance)
    total_score += calculate_attribute_match_points(mentor_row["Roles"], mentee_row["Roles"], 15)
    total_score += calculate_attribute_match_points(mentor_row["Topics"], mentee_row["Topics"], 12)
    total_score += calculate_attribute_match_points(mentor_row["Industry"], mentee_row["Industry"], 10)
    
    # Company stage matching (exclude "No Preference" as requested)
    total_score += calculate_attribute_match_points(
        mentor_row["Company Stage"], mentee_row["Company Stage"], 10, exclude_no_preference=True
    )
    
    total_score += calculate_attribute_match_points(
        mentor_row["In-Person Meeting Location"], mentee_row["In-Person Meeting Location"], 5
    )
    
    # Important attributes exact matching bonus
    for attribute_column in important_attribute_columns:
        mentor_value = mentor_row.get(attribute_column, "No Preference")
        mentee_value = mentee_row.get(attribute_column, "No Preference")
        if (mentor_value != "No Preference" and mentee_value != "No Preference" and mentor_value == mentee_value):
            total_score += 15.0
    
    # Time slot preference bonus (reduced weight)
    total_score += calculate_time_slot_bonus(mentor_row["Time Slot Preference"], mentee_row["Time Slot Preference"])
    
    # Ensure minimum viable score (prevents extremely low scores)
    return max(total_score, 50)

# ================
# PAIRWISE COMPATIBILITY CALCULATION
# ================

print("Computing compatibility scores for all mentor-mentee pairs...")

# Get all mentor and mentee indices for iteration
mentor_indices = mentors.index.tolist()
mentee_indices = mentees.index.tolist()
pair_compatibility_scores = {}

# Calculate compatibility score for every possible mentor-mentee combination
for mentor_idx in mentor_indices:
    for mentee_idx in mentee_indices:
        score = compute_overall_match_score(
            mentors.loc[mentor_idx], mentees.loc[mentee_idx], mentor_idx, mentee_idx
        )
        
        # Only store viable matches (ignore impossible pairings)
        if score > -np.inf:
            pair_compatibility_scores[(mentor_idx, mentee_idx)] = score

print(f"Found {len(pair_compatibility_scores)} viable mentor-mentee pairs")

if not pair_compatibility_scores:
    raise ValueError("No valid matches found under current constraints. Check your data and business rules.")

# ================
# OPTIMIZATION SETUP (LINEAR PROGRAMMING)
# ================

print("Setting up optimization problem to maximize matches...")

# Initialize OR-Tools solver for Mixed Integer Linear Programming
solver = pywraplp.Solver.CreateSolver('SCIP')
if not solver:
    raise RuntimeError("Failed to initialize OR-Tools solver.")

# Create binary decision variables: x[mentor_i, mentee_j] = 1 if matched, 0 otherwise
matching_variables = {}
for (mentor_idx, mentee_idx) in pair_compatibility_scores.keys():
    var_name = f"x_{mentor_idx}_{mentee_idx}"
    matching_variables[(mentor_idx, mentee_idx)] = solver.BoolVar(var_name)

# ================
# BUSINESS CONSTRAINTS
# ================

print("Adding business constraints...")

# CONSTRAINT 1: Each mentee can have at most 1 mentor
# Ensures no mentee is overwhelmed with multiple mentoring relationships
for mentee_idx in mentee_indices:
    potential_matches = [
        matching_variables[(mentor_idx, mentee_idx)] 
        for mentor_idx in mentor_indices 
        if (mentor_idx, mentee_idx) in matching_variables
    ]
    if potential_matches:
        solver.Add(solver.Sum(potential_matches) <= 1)

# CONSTRAINT 2: Each mentor can have at most 2 mentees
# Prevents mentor overload while allowing some mentors to help multiple people
for mentor_idx in mentor_indices:
    potential_matches = [
        matching_variables[(mentor_idx, mentee_idx)] 
        for mentee_idx in mentee_indices 
        if (mentor_idx, mentee_idx) in matching_variables
    ]
    if potential_matches:
        solver.Add(solver.Sum(potential_matches) <= 2)

# ================
# OPTIMIZATION OBJECTIVE
# ================

print("Setting up multi-objective optimization...")

# Count total number of matches made
total_matches = solver.Sum([
    matching_variables[(mentor_idx, mentee_idx)] 
    for (mentor_idx, mentee_idx) in matching_variables.keys()
])

# Sum total quality across all matches
total_quality = solver.Sum([
    score * matching_variables[(mentor_idx, mentee_idx)]
    for (mentor_idx, mentee_idx), score in pair_compatibility_scores.items()
])

# WEIGHTED OBJECTIVE: Prioritize match count heavily, then optimize quality
# This encourages finding more matches even if individual quality is lower
MATCH_COUNT_WEIGHT = 1000  # High weight to maximize number of people matched
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

# ================
# EXTRACT AND ANALYZE RESULTS
# ================

# Extract which mentor-mentee pairs were selected in the optimal solution
mentor_to_mentees_mapping = defaultdict(list)
for (mentor_idx, mentee_idx), decision_variable in matching_variables.items():
    if decision_variable.solution_value() > 0.5:  # Variable is set to 1 (matched)
        mentor_to_mentees_mapping[mentor_idx].append(mentee_idx)

# Calculate match statistics
total_matched = sum(len(mentees_list) for mentees_list in mentor_to_mentees_mapping.values())
matched_new = sum(1 for mentor_idx, mentee_list in mentor_to_mentees_mapping.items()
                 for mentee_idx in mentee_list
                 if not mentees.loc[mentee_idx, "Has_Previous_Matches"])

print(f"✅ Optimal solution found!")
print(f"📊 Matched {total_matched}/{len(mentees)} mentees ({total_matched/len(mentees)*100:.1f}%)")
print(f"🆕 New mentees matched: {matched_new}")

# ================
# OUTPUT GENERATION WITH INDIVIDUAL OPEN ANSWERS
# ================

# Define columns to include in output CSV (includes individual Open Answers for QA)
output_columns = [
    "Offset", "Years of Experience", "Roles", "Topics", "Industry", 
    "Company Stage", "In-Person Meeting Location", "Time Slot Preference", 
    "Goal_Topic", "Has_Previous_Matches", 
    "Open Answer"  # INDIVIDUAL'S OWN ANSWER (FIXED)
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

output_data_rows = []
used_submission_ids = set()  # Prevent duplicate entries

print("Generating output with individual Open Answers...")

# Process each mentor and their assigned mentees
for mentor_idx in sorted(mentor_to_mentees_mapping.keys(), key=get_mentor_sort_key):
    mentor_data = mentors.loc[mentor_idx]
    mentor_submission_id = mentor_data.get("Mentor Submission ID", "")
    
    # Only process if this mentor hasn't been added yet (prevent duplicates)
    if mentor_submission_id and mentor_submission_id not in used_submission_ids:
        used_submission_ids.add(mentor_submission_id)
        
        # Create mentor row with MENTOR'S OWN Open Answer
        mentor_row = {
            "Row Type": "Mentor", 
            "Name": mentor_data.get("Full Name", ""),
            "Email": mentor_data.get("Coda Email", ""), 
            "Submission ID": mentor_submission_id,
            "Match Score": ""  # No score for mentor rows
        }
        
        # Fill in all columns for mentor row
        for column in output_columns:
            if column == "Open Answer":
                # FIXED: Show mentor's own Open Answer
                mentor_row[column] = mentor_data.get("Open Answer", "")
            elif column in mentor_data.index:
                mentor_row[column] = format_set_for_output(mentor_data[column])
            else:
                mentor_row[column] = "N/A"
        
        output_data_rows.append(mentor_row)
        
        # Process this mentor's assigned mentees (sorted by match score)
        assigned_mentees = sorted(
            mentor_to_mentees_mapping[mentor_idx], 
            key=lambda x: get_mentee_sort_key(mentor_idx, x)
        )
        
        for mentee_idx in assigned_mentees:
            mentee_data = mentees.loc[mentee_idx]
            mentee_submission_id = mentee_data.get("Mentee Submission ID", "")
            
            # Only process if this mentee hasn't been added yet (prevent duplicates)
            if mentee_submission_id and mentee_submission_id not in used_submission_ids:
                used_submission_ids.add(mentee_submission_id)
                match_score = pair_compatibility_scores[(mentor_idx, mentee_idx)]
                
                # Create mentee row with MENTEE'S OWN Open Answer
                mentee_row = {
                    "Row Type": "Mentee", 
                    "Name": mentee_data.get("Full Name", ""),
                    "Email": mentee_data.get("Coda Email", ""), 
                    "Submission ID": mentee_submission_id,
                    "Match Score": round(match_score, 2)
                }
                
                # Fill in all columns for mentee row
                for column in output_columns:
                    if column == "Open Answer":
                        # FIXED: Show mentee's own Open Answer
                        mentee_row[column] = mentee_data.get("Open Answer", "")
                    else:
                        mentee_row[column] = format_set_for_output(mentee_data[column])
                
                output_data_rows.append(mentee_row)

# ================
# FINAL OUTPUT PROCESSING AND SAVE
# ================

# Create final DataFrame and remove any remaining duplicates
final_output_df = pd.DataFrame(output_data_rows)
final_output_df = final_output_df.drop_duplicates(subset=['Submission ID'], keep='first')

# Save results to CSV
print("Saving results...")
final_output_df.to_csv(OUTPUT_CSV, index=False)

# ================
# SUMMARY AND COMPLETION
# ================

print(f"✅ Matching complete! Results saved to: {OUTPUT_CSV}")
print(f"📁 Final output contains {len(final_output_df)} rows")
print(f"📝 Individual Open Answer data included for each person's row")

# Display final scoring breakdown for transparency
print("\n" + "="*70)
print("SCORING BREAKDOWN SUMMARY")
print("="*70)
print("🆕 NEW MENTEE BONUS: +50 points (prioritize first-time participants)")
print("🎯 GOAL ALIGNMENT: Up to 75 points (enhanced text similarity matching)")
print("🎯 TOPIC MATCH BONUS: Up to 30 points (same goal category)")
print("💼 EXPERIENCE GAP: 50-160 points (mentor must have more experience)")
print("🌍 TIMEZONE FLEXIBILITY: -40 to +30 points (soft constraint)")
print("👔 SHARED ROLES: 15 points per shared role")
print("📚 SHARED TOPICS: 12 points per shared topic")
print("🏭 SHARED INDUSTRY: 10 points per shared industry")
print("🏢 SHARED COMPANY STAGE: 10 points per stage (excludes 'No Preference')")
print("📍 SHARED LOCATIONS: 5 points per shared location")
print("⭐ IMPORTANT ATTRIBUTES: 15 points per exact match")
print("⏰ TIME SLOT MATCH: Up to 15 points (reduced weighting)")
print("="*70)
print("🔧 TECHNICAL: No PyTorch/sklearn dependencies - pure text matching")
print("📋 QA READY: Individual Open Answers for each person's row")
print("🎯 ALGORITHM: Mixed Integer Linear Programming for optimal assignments")
print("="*70)

print("\n🎉 Ready for manual QA review!")
