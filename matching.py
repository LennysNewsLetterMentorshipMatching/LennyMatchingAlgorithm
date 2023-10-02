import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

#define max mentees/Users/ijac/side-projects/lenny_mentorship/Mentee_for_matching.csv /Users/ijac/side-projects/lenny_mentorship/Mentor_for_matching.csv
max_mentees_per_mentor = 2

#load data
mentors = pd.read_csv('Mentor_for_matching.csv')
mentees = pd.read_csv('Mentee_for_matching.csv')


#Clean DF
#define function to merge columns with same names together
def same_merge(x): return ','.join(x[x.notnull()].astype(str))

#define new DataFrame that merges columns with same names together
mentees = mentees.groupby(level=0, axis=1).apply(lambda x: x.apply(same_merge, axis=1))

print("mentor columns:", mentors.columns.values)
print("mentee columns:", mentees.columns.values)

mentors_flitered = mentors.filter(items=["Email",
                 "Offset",
                 'In-Person Meeting Location',
                 "Avg Year of YOE",
                 'Roles',
                 'Industry',
                 'Company Stage',
                 'Topics',
                 'Most Important Attribute',
                 'Created on'
                ])

mentees_flitered = mentees.filter(items=["Email",
                 "Offset",
                 'In-Person Meeting Location',
                 "Avg Year of YOE",
                 'Roles',
                 'Industry',
                 'Company Stage',
                 'Topics',
                 'Most Important Attribute',
                 'Created on'
                ])

print("mentor filter columns:", mentors_flitered.columns.values)
print("mentee filter columns:", mentees_flitered.columns.values)

# # display(mentors_flitered) 
# # display(mentees_flitered)

# #Input comma seperated list of value
# #Output list of values with whitespace stipped off
def clean_multiselect(x):
    if isinstance(x, str):
        return list(map(str.strip,x.split(',')))
    else:
        return []
      
# #Input Dataframe and multi-select field to Binarize
def MultiLableBinarize_df(input_frame, collumn_name):
    nested_list = list(map(clean_multiselect,input_frame[collumn_name].to_list()))
    mlb = MultiLabelBinarizer()
    mlb_df = pd.DataFrame(mlb.fit_transform(nested_list), columns=mlb.classes_)
    bigger = pd.concat([input_frame,mlb_df],axis=1)
    return bigger
  
  
class multiSelect:
    def __init__(self, data = ['empty']):
        if isinstance(data, str):
            self.data = clean_multiselect(data)
        else:
            self.data = data
    def __repr__(self):
        return repr(self.data)

class distanceEstimator:
    def __init__(self, mentor_mentee_question_mapping = []):
        self.mentor_mentee_question_mapping = mentor_mentee_question_mapping
        
    def multiSelectDistance(self,row,mentee_selection,mentor_selection):
        distance_score = 0
        matched = []
        if isinstance(mentee_selection,list) and isinstance(mentor_selection,list):
            for selection in mentee_selection:
                if selection in mentor_selection:
                    distance_score = distance_score - 10
                    matched.append(selection)
        return distance_score, matched
    
    def yoeDistance(self, mentor_yoe, mentee_yoe):
        difference = mentor_yoe - mentee_yoe
        if difference >= 8:
            return 50
        elif 4 <= difference < 8:
            return 100
        elif 2 <= difference < 4:
            return 160
        else:  # difference <= 1 or mentor_yoe <= mentee_yoe
            return -1000
    
    def _estimateDistance(self, row):
        matched = []
        distance_score = 1000
        
        # # List of attribute names to check for a match with "Most Important Attribute"
        # attribute_list = ["Offset", 'In-Person Meeting Location', "Avg Year of YOE",
        #               'Roles', 'Industry', 'Company Stage', 'Topics']
    
        # # Get the mentor's and mentee's "Most Important Attribute"
        # mentor_important_attribute = row["Most Important Attribute-mentor"]
        # mentee_important_attribute = row["Most Important Attribute-mentee"]
        
        # # Determine the attribute to be given higher weight
        # important_attribute = None
        # if mentor_important_attribute == mentee_important_attribute and mentor_important_attribute in attribute_list:
        #     important_attribute = mentor_important_attribute
    
    
        # Iterate over the mapping of questions
        for mapping in self.mentor_mentee_question_mapping:
          if mapping['mentee_question'] == mapping['mentor_question']:
            mentee_question = mapping['mentee_question'] + "-mentee"
            mentor_question = mapping['mentee_question'] + '-mentor'
          else:
            mentee_question = mapping['mentee_question']
            mentor_question = mapping['mentor_question']
          if mapping['question_type'] == 'multi-select':
            mentee_selection = row[mentee_question].data
            mentor_selection = row[mentor_question].data

            distance_score_temp, matched_temp = self.multiSelectDistance(row,mentee_selection,mentor_selection)

            # Adjust the weight if the current attribute is the important one -- TBD
            # if mapping['mentee_question'] == important_attribute:
            #     distance_score -= distance_score_temp * 10 * mapping['question_weight']
            # else:
            #     distance_score -= distance_score_temp * mapping['question_weight']
            
            # matched = matched + matched_temp
            distance_score = distance_score + distance_score_temp*mapping['question_weight']
            matched = matched + matched_temp
        
        # Adding the YOE scoring
        mentor_yoe = float(row["Avg Year of YOE-mentor"])
        mentee_yoe = float(row["Avg Year of YOE-mentee"])
        distance_score -= self.yoeDistance(mentor_yoe, mentee_yoe)
    
    
        return distance_score, multiSelect(matched)

    def estimateDistance(self, row):
        distance_score, matched = self._estimateDistance(row)
        return distance_score

    def matched(self,row):
        distance_score, matched = self._estimateDistance(row)
        return matched

print("mentor column value:", mentors_flitered.columns.values)
print("mentee column value:", mentees_flitered.columns.values)

mentor_mentee_question_mapping = [{'mentee_question':'Offset',
                                   'mentor_question':'Offset',
                                   'question_type': 'multi-select',
                                   'question_weight': 2,},
                                  {'mentee_question':'In-Person Meeting Location',
                                   'mentor_question':'In-Person Meeting Location',
                                   'question_type': 'multi-select',
                                   'question_weight': 1,},
                                  {'mentee_question':'Roles',
                                   'mentor_question':'Roles',
                                   'question_type': 'multi-select',
                                   'question_weight': 8,},
                                  {'mentee_question':'Industry',
                                   'mentor_question':'Industry',
                                   'question_type': 'multi-select',
                                   'question_weight': 6,},
                                  {'mentee_question':'Company Stage',
                                   'mentor_question':'Company Stage',
                                   'question_type': 'multi-select',
                                   'question_weight': 5,},
                                  {'mentee_question':'Topics',
                                   'mentor_question':'Topics',
                                   'question_type': 'multi-select',
                                   'question_weight': 7,}
                                  ]

for mapping in mentor_mentee_question_mapping:
  if mapping['question_type'] == 'multi-select':
    mentees_flitered[mapping['mentee_question']] = mentees_flitered[mapping['mentee_question']].apply(multiSelect)
    mentors_flitered[mapping['mentor_question']] = mentors_flitered[mapping['mentor_question']].apply(multiSelect)


combined = mentors_flitered.join(mentees_flitered,how='cross',lsuffix='-mentor',rsuffix='-mentee')
# checking 
# combined.to_csv('combined.csv', index=False) 


dE = distanceEstimator(mentor_mentee_question_mapping)
combined['distance_score'] = combined.apply(dE.estimateDistance, axis = 'columns')
# combined.to_csv('combined_score.csv', index=False)
combined['matched_criteria'] = combined.apply(dE.matched, axis = 'columns')
# combined.to_csv('combined_apply.csv', index=False) 
combined = combined.sort_values(by=['distance_score'])

matched_mentors = {}
matched_mentees = {}
matched_list = []

mentee_id = 'Email-mentee'
mentor_id = 'Email-mentor'

for index, row in combined.iterrows():
    #print(type(row['Id-mentor']))
    #print(type(row['Id-mentee']))
    #print(row['Id-mentee'] == row['Id-mentor'])
    # print('Check if this prints correctly', row['matched_criteria'])  # Check if this prints correctly
    
    # Extract YOE for mentor and mentee
    mentor_yoe = float(row["Avg Year of YOE-mentor"])
    mentee_yoe = float(row["Avg Year of YOE-mentee"])
    # Skip this iteration if mentor's YOE is less than mentee's YOE
    if mentor_yoe <= mentee_yoe:
        continue
    # Existing conditions and logic    
    if row[mentor_id] not in matched_mentors:
        matched_mentors[row[mentor_id]] = 0
    if row[mentee_id] not in matched_mentees:
        matched_mentees[row[mentee_id]] = 0
    if matched_mentors[row[mentor_id]] >= max_mentees_per_mentor:
        continue
    if matched_mentees[row[mentee_id]] >=1:
        continue
    if row[mentee_id] == row[mentor_id]:
        #print('skipped, matching to self')
        continue
    matched_mentors[row[mentor_id]] = matched_mentors[row[mentor_id]] + 1
    matched_mentees[row[mentee_id]] = matched_mentees[row[mentee_id]] + 1
    matched_list.append({mentor_id:row[mentor_id],mentee_id:row[mentee_id], 'distance_score':row['distance_score'], 'matched':str(row['matched_criteria'])})

results = pd.DataFrame(matched_list)
results.to_csv('matched_list.csv', index=False)
reuslts_wide = results.join(mentors_flitered.set_index('Email'),on = mentor_id, rsuffix='-mentor').join(mentees_flitered.set_index('Email'),on = mentee_id,lsuffix='-mentor', rsuffix='-mentee')

results.to_csv('matched.csv', index=False)
reuslts_wide.to_csv('matched_wide.csv', index=False)
