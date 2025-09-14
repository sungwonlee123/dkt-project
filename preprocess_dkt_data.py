import pandas as pd
import json
import numpy as np
from pathlib import Path

def load_concept_relationships(json_path):
    """
    Load and process the concept relationship data.
    Returns a dictionary mapping concept IDs to their metadata and relationships.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        concept_data = json.load(f)
    
    concept_relationships = {}
    for _, rel in concept_data.items():
        from_concept = rel['fromConcept']
        to_concept = rel['toConcept']
        
        # Store relationship information
        if from_concept['id'] not in concept_relationships:
            concept_relationships[from_concept['id']] = {
                'name': from_concept['name'],
                'semester': from_concept['semester'],
                'chapter': from_concept['chapter'],
                'prerequisites': set(),
                'postrequisites': set()
            }
        
        if to_concept['id'] not in concept_relationships:
            concept_relationships[to_concept['id']] = {
                'name': to_concept['name'],
                'semester': to_concept['semester'],
                'chapter': to_concept['chapter'],
                'prerequisites': set(),
                'postrequisites': set()
            }
        
        # Add relationship
        concept_relationships[from_concept['id']]['postrequisites'].add(to_concept['id'])
        concept_relationships[to_concept['id']]['prerequisites'].add(from_concept['id'])
    
    return concept_relationships

def load_irt_data(item_irt_path, user_irt_path):
    """
    Load IRT (Item Response Theory) data for both items and users.
    Returns two dictionaries with IRT parameters.
    """
    # Load item IRT data
    with open(item_irt_path, 'r', encoding='utf-8') as f:
        item_irt_raw = json.load(f)
    
    # Process item IRT data
    item_irt = {}
    if 'properties' in item_irt_raw:  # Skip the schema part
        for record in item_irt_raw.get('data', []):
            item_irt[record['testID']] = {
                'difficulty': record.get('difficultyLevel', 0),
                'discrimination': record.get('discriminationLevel', 1),
                'knowledge_tag': record.get('knowledgeTag', '')
            }
    
    # Load user IRT data
    with open(user_irt_path, 'r', encoding='utf-8') as f:
        user_irt_raw = json.load(f)
    
    # Process user IRT data
    user_irt = {}
    if 'properties' in user_irt_raw:  # Skip the schema part
        for record in user_irt_raw.get('data', []):
            user_irt[record['userID']] = {
                'ability': record.get('abilityLevel', 0),
                'consistency': record.get('consistencyLevel', 1)
            }
    
    return item_irt, user_irt

def process_student_responses(csv_path, concept_relationships, item_irt, user_irt):
    """
    Process student response data and create a dataset suitable for DKT modeling.
    """
    # Initialize lists to store processed data
    processed_data = []
    
    with open(csv_path, 'r') as f:
        current_user = None
        problem_ids = []
        responses = []
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if ',' not in line:  # This is a user ID
                # Process previous user's data if exists
                if current_user and problem_ids:
                    for i, (prob_id, resp) in enumerate(zip(problem_ids, responses)):
                        item_data = item_irt.get(prob_id, {})
                        user_data = user_irt.get(current_user, {})
                        
                        entry = {
                            'user_id': current_user,
                            'item_id': prob_id,
                            'correct': resp,
                            'position': i,  # Sequential position in the session
                            'difficulty': item_data.get('difficulty', 0),
                            'discrimination': item_data.get('discrimination', 1),
                            'knowledge_tag': item_data.get('knowledge_tag', ''),
                            'user_ability': user_data.get('ability', 0),
                            'user_consistency': user_data.get('consistency', 1)
                        }
                        
                        # Add prerequisite information if available
                        if item_data.get('knowledge_tag') in concept_relationships:
                            concept = concept_relationships[item_data['knowledge_tag']]
                            entry.update({
                                'concept_name': concept['name'],
                                'semester': concept['semester'],
                                'chapter': concept['chapter'],
                                'prerequisites': list(concept['prerequisites']),
                                'postrequisites': list(concept['postrequisites'])
                            })
                        
                        processed_data.append(entry)
                
                # Start new user
                current_user = line
                problem_ids = []
                responses = []
            else:
                # This line contains either problem IDs or responses
                values = line.split(',')
                if problem_ids:  # If we already have problem IDs, these must be responses
                    responses = [int(x) for x in values]
                else:  # These are problem IDs
                    problem_ids = values
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    # Sort by user_id and position
    df = df.sort_values(['user_id', 'position'])
    
    # Add sequence-related features
    df['skill_id'] = df['knowledge_tag']  # Use knowledge tag as skill ID
    
    # Calculate running statistics per user
    df['cumulative_attempts'] = df.groupby('user_id').cumcount() + 1
    df['running_correct'] = df.groupby('user_id')['correct'].cumsum()
    df['running_accuracy'] = df['running_correct'] / df['cumulative_attempts']
    
    return df

def main():
    base_path = Path('/Users/iseong-won/dkt project/aihub')
    
    # File paths
    concept_path = base_path / '[라벨]수학 지식체계 데이터 세트_210611.json'
    train_path = base_path / 'Training/[라벨]i-scream_train.csv'
    item_irt_path = base_path / 'Training/[라벨]2_문항IRT_rule_20210210164012.json'
    user_irt_path = base_path / 'Training/[라벨]3_응시자IRT_rule_20210210164059.json'
    
    # Load concept relationships
    print("Loading concept relationships...")
    concept_relationships = load_concept_relationships(concept_path)
    
    # Load IRT data
    print("Loading IRT data...")
    item_irt, user_irt = load_irt_data(item_irt_path, user_irt_path)
    
    # Process student responses
    print("Processing student responses...")
    dkt_dataset = process_student_responses(train_path, concept_relationships, item_irt, user_irt)
    
    # Save processed dataset
    output_path = base_path.parent / 'processed_dkt_data.csv'
    dkt_dataset.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")

if __name__ == '__main__':
    main()
