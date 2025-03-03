import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
from IPython.display import FileLink
from flair.models import TextClassifier
from flair.data import Sentence

# Load Flair sentiment model
sentiment_model = TextClassifier.load('en-sentiment')

# Load mentor and mentee data
mentors = pd.read_csv('MentorSubFall2025.csv')
mentees = pd.read_csv('MenteeSubFall2025.csv')

# Load previously matched pairs
previous_matches = pd.read_csv('matched_pairs2025_pulp.csv', sep='\t')

# Clean and filter mentor and mentee data
mentors_filtered = mentors.filter(items=["Email", "Offset", "Avg Year of YOE", "Important Attribute - First", 
                                         "Important Attribute - Second", "Important Attribute - Third", "Topics", "Open Answer"])
mentees_filtered = mentees.filter(items=["Email", "Offset", "Avg Year of YOE", "Important Attribute - First", 
                                         "Important Attribute - Second", "Important Attribute - Third", "Topics", "Open Answer"])

# Function to process comma-separated topics into a list
def clean_multiselect(x):
    if isinstance(x, str):
        return [item.strip() for item in x.split(',')]
    else:
        return []

# Apply the cleaning function to the topics column
mentors_filtered['Topics'] = mentors_filtered['Topics'].apply(clean_multiselect)
mentees_filtered['Topics'] = mentees_filtered['Topics'].apply(clean_multiselect)

# Function to get sentiment score using Flair
def get_sentiment_score(text):
    if pd.isna(text):
        return 0
    sentence = Sentence(text)
    sentiment_model.predict(sentence)
    score = sentence.labels[0].score
    return score if sentence.labels[0].value == 'POSITIVE' else -score

# Apply sentiment analysis to 'Open Answer' column
mentors_filtered['Sentiment'] = mentors_filtered['Open Answer'].apply(get_sentiment_score)
mentees_filtered['Sentiment'] = mentees_filtered['Open Answer'].apply(get_sentiment_score)

# Define a function to calculate matching score
def calculate_score(mentor, mentee):
    score = 1000
    yoe_diff = mentor['Avg Year of YOE'] - mentee['Avg Year of YOE']
    offset_diff = abs(mentor['Offset'] - mentee['Offset'])
    sentiment_diff = abs(mentor['Sentiment'] - mentee['Sentiment'])

    # Define attribute weights
    attribute_weights = {
        'No Preference': 0,
        'Years of Experience': 50,
        'Role': 40,
        'Industry': 30,
        'Company Stage': 20,
        'Topic': 10,
        'In-Person Meeting': 5,
        'Shared Identity': 5
    }

    # Calculate score based on important attributes
    for attr in ["Important Attribute - First", "Important Attribute - Second", "Important Attribute - Third"]:
        weight = attribute_weights.get(mentor[attr].strip(), 0)
        if weight > 0 and mentor[attr].strip() == mentee[attr].strip():
            score += weight

    # Calculate score based on years of experience
    if yoe_diff > 8:
        score += 50
    elif 4 <= yoe_diff <= 8:
        score += 100
    elif 2 <= yoe_diff <= 3:
        score += 160
    elif yoe_diff <= 0:
        score -= 1000

    # Penalty for offset difference
    score -= offset_diff * 10

    # Calculate score based on matching topics
    common_topics = set(mentor['Topics']).intersection(set(mentee['Topics']))
    score += len(common_topics) * 20

    # Add score based on sentiment similarity
    score += (1 - sentiment_diff) * 50

    return score

# Create LP problem instance
prob = LpProblem("Mentor_Mentee_Matching", LpMaximize)

# Create decision variables for mentor-mentee pairs
mentor_mentee_pairs = [(mentor_index, mentee_index) for mentor_index in range(len(mentors_filtered)) 
                       for mentee_index in range(len(mentees_filtered))]
pair_vars = LpVariable.dicts("Pair", mentor_mentee_pairs, cat='Binary')

# Define the objective function
prob += lpSum(pair_vars[mentor_index, mentee_index] * calculate_score(mentors_filtered.iloc[mentor_index], mentees_filtered.iloc[mentee_index]) 
              for mentor_index, mentee_index in mentor_mentee_pairs)

# Add constraints
for mentee_index in range(len(mentees_filtered)):
    prob += lpSum(pair_vars[mentor_index, mentee_index] for mentor_index in range(len(mentors_filtered))) <= 1

for mentor_index in range(len(mentors_filtered)):
    prob += lpSum(pair_vars[mentor_index, mentee_index] for mentee_index in range(len(mentees_filtered))) <= 2
    prob += lpSum(pair_vars[mentor_index, mentee_index] for mentee_index in range(len(mentees_filtered))) >= 1

for mentor_index, mentee_index in mentor_mentee_pairs:
    mentor_email = mentors_filtered.iloc[mentor_index]['Email']
    mentee_email = mentees_filtered.iloc[mentee_index]['Email']
    
    if mentor_email == mentee_email:
        prob += pair_vars[mentor_index, mentee_index] == 0
    
    offset_diff = abs(mentors_filtered.iloc[mentor_index]['Offset'] - mentees_filtered.iloc[mentee_index]['Offset'])
    prob += pair_vars[mentor_index, mentee_index] * offset_diff <= 2

    if not previous_matches[(previous_matches['Mentor'] == mentor_email) & (previous_matches['Mentee'] == mentee_email)].empty:
        prob += pair_vars[mentor_index, mentee_index] == 0

# Solve the linear programming problem
prob.solve()

# Extract matched pairs from the solved LP problem
matched_pairs = {}
for (mentor_index, mentee_index), var in pair_vars.items():
    if var.value() == 1:
        mentor_email = mentors_filtered.loc[mentor_index, 'Email']
        mentee_email = mentees_filtered.loc[mentee_index, 'Email']
        matched_pairs.setdefault(mentor_email, []).append(mentee_email)

# Create DataFrame for matched pairs
matched_df = pd.DataFrame([(mentee, mentor) for mentor, mentees in matched_pairs.items() for mentee in mentees],
                           columns=['Mentee', 'Mentor'])

# Save matched pairs to CSV
matched_df.to_csv('matched_pairs2025_pulp_modified.csv', index=False, sep='\t')

# Display the CSV files for download
display(FileLink('matched_pairs2025_pulp_modified.csv'))
