import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from ortools.linear_solver import pywraplp

# === File paths ===
MENTORS_CSV = 'Mentor-Original-Submissions.csv'
MENTEES_CSV = 'Mentee-Original-Submissions.csv'

# === Load datasets ===
mentors = pd.read_csv(MENTORS_CSV)
mentees = pd.read_csv(MENTEES_CSV)

# === Preprocess multi-select columns to sets for efficient intersection ===
def to_set(val):
    if pd.isna(val):
        return set()
    return set(s.strip() for s in str(val).split(','))

multi_select_cols = ["Roles", "Topics", "Industry", "Company Stage", "In-Person Meeting Location"]
for col in multi_select_cols:
    mentors[col] = mentors[col].apply(to_set)
    mentees[col] = mentees[col].apply(to_set)

# Convert numeric fields
for col in ["Offset", "Avg Year of YOE"]:
    mentors[col] = pd.to_numeric(mentors[col], errors='coerce')
    mentees[col] = pd.to_numeric(mentees[col], errors='coerce')

# Fill missing important attributes and time slot prefs
important_attrs = ["Important Attribute - First", "Important Attribute - Second", "Important Attribute - Third"]
for attr in important_attrs:
    mentors[attr] = mentors[attr].fillna("No Preference").str.strip()
    mentees[attr] = mentees[attr].fillna("No Preference").str.strip()

mentors["Time Slot Preference"] = mentors["Time Slot Preference"].fillna("No Preference").str.strip()
mentees["Time Slot Preference"] = mentees["Time Slot Preference"].fillna("No Preference").str.strip()

# === Sentiment analysis setup ===
sia = SentimentIntensityAnalyzer()

def sentiment_score(text):
    if pd.isna(text) or text.strip() == "":
        return 0.0
    return sia.polarity_scores(text)["compound"]

mentors["Sentiment"] = mentors["Open Answer"].fillna("").apply(sentiment_score)
mentees["Sentiment"] = mentees["Open Answer"].fillna("").apply(sentiment_score)

# === Scoring function as per your specification ===
def compute_match_score(m, n):
    # Time zone proximity constraint
    if pd.isna(m["Offset"]) or pd.isna(n["Offset"]) or abs(m["Offset"] - n["Offset"]) > 2:
        return -np.inf

    # Mentor YOE must be greater
    yoe_diff = m["Avg Year of YOE"] - n["Avg Year of YOE"]
    if pd.isna(yoe_diff) or yoe_diff <= 0:
        return -np.inf

    score = 1000
    if 2 <= yoe_diff <= 3:
        score += 160
    elif 4 <= yoe_diff <= 8:
        score += 100
    elif yoe_diff > 8:
        score += 50

    weights = {
        "Roles": 8,
        "Topics": 7,
        "Industry": 6,
        "Company Stage": 5,
        "Offset": 2,
        "In-Person Meeting Location": 1,
    }

    # Subtract 10 * weight for each matched attribute in intersection
    for attr in multi_select_cols:
        score -= len(m[attr].intersection(n[attr])) * 10 * weights[attr]

    # Offset exact match bonus (subtract penalty if exact match)
    if m["Offset"] == n["Offset"]:
        score -= 1 * 10 * weights["Offset"]

    # Important Attribute - no preference logic (no penalty or bonus)
    for attr in important_attrs:
        m_val = m.get(attr, "No Preference").strip()
        n_val = n.get(attr, "No Preference").strip()
        if (m_val == n_val == "No Preference") or (m_val == n_val):
            # no penalty
            pass

    # Sentiment alignment penalty
    sentiment_diff = abs(m["Sentiment"] - n["Sentiment"])
    score -= sentiment_diff * 10

    # Time Slot Preference penalty if they differ and neither is no preference
    if m["Time Slot Preference"] != "No Preference" and n["Time Slot Preference"] != "No Preference":
        if m["Time Slot Preference"] != n["Time Slot Preference"]:
            score -= 5

    return score

# === Compute scores for all pairs ===
mentor_ids = mentors.index.tolist()
mentee_ids = mentees.index.tolist()
pair_scores = {}

print("Computing pairwise scores...")
for m_id in mentor_ids:
    m = mentors.loc[m_id]
    for n_id in mentee_ids:
        n = mentees.loc[n_id]
        score = compute_match_score(m, n)
        if score > -np.inf:
            pair_scores[(m_id, n_id)] = score

if not pair_scores:
    raise ValueError("No valid matches found with given constraints.")

# === Setup OR-Tools solver ===
solver = pywraplp.Solver.CreateSolver('SCIP')
if not solver:
    raise RuntimeError("OR-Tools solver not created.")

# Decision variables for matching pairs
x = {}
for (m_id, n_id) in pair_scores.keys():
    x[(m_id, n_id)] = solver.BoolVar(f"x_{m_id}_{n_id}")

# Constraints
# Each mentee assigned to at most one mentor
for n_id in mentee_ids:
    solver.Add(solver.Sum(x[(m_id, n_id)] for m_id in mentor_ids if (m_id, n_id) in x) <= 1)

# Each mentor assigned to at most two mentees
for m_id in mentor_ids:
    solver.Add(solver.Sum(x[(m_id, n_id)] for n_id in mentee_ids if (m_id, n_id) in x) <= 2)

# Objective: maximize total matching score
objective = solver.Objective()
for (m_id, n_id), score in pair_scores.items():
    objective.SetCoefficient(x[(m_id, n_id)], score)
objective.SetMaximization()

print("Solving the matching optimization problem...")
status = solver.Solve()
if status != pywraplp.Solver.OPTIMAL:
    raise RuntimeError("No optimal solution found.")

print("Optimal matching found.")

# === Extract matching results ===
matches = []
for (m_id, n_id), var in x.items():
    if var.solution_value() > 0.5:
        matches.append({
            "Mentor Name": mentors.loc[m_id, "Full Name"],
            "Mentor Email": mentors.loc[m_id, "Coda Email"],
            "Mentee Name": mentees.loc[n_id, "Full Name"],
            "Mentee Email": mentees.loc[n_id, "Coda Email"],
            "Match Score": round(pair_scores[(m_id, n_id)], 2)
        })

matches_df = pd.DataFrame(matches).sort_values(by="Match Score", ascending=False)

# === Save output CSV ===
OUTPUT_CSV = "mentor_mentee_best_matches_with_scores.csv"
matches_df.to_csv(OUTPUT_CSV, index=False)
print(f"Matching results saved to '{OUTPUT_CSV}'.")

# === Explanation (printed to console) ===
print("""
Explanation:
- Each pair starts with a base score of 1000.
- The mentor's Years of Experience (YOE) must be greater than the mentee's.
- YOE difference adds 160 points if between 2-3 years, 100 if 4-8, 50 if 8+.
- Matches on Roles, Topics, Industry, Company Stage, Offset, and In-Person Meeting Location reduce the score by 10 times their attribute weights per match.
- Weights used: Roles=8, Topics=7, Industry=6, Company Stage=5, Offset=2, In-Person Meeting=1.
- Mentor and mentee must be within 2 time zones.
- Differences in sentiment reduce the score by 10 times the absolute difference.
- Time Slot Preference differs: -5 points.
- Mentors can have up to 2 mentees; mentees only 1 mentor.
- Final CSV includes all matched pairs with their computed scores.
""")
