import json
from openai import OpenAI
import os
from math import floor 
    

def find_number_of_questions(file_path):
    count = 0
    data_points =0
    with open(file_path, "r") as file:
        data = json.load(file)
        for entry in data:
            question_count = floor(entry.get('size', )/1000)+1
            if question_count>4:
                question_count=4
            count += question_count
            data_points += 1
    return count, data_points

key = ""

openai_client = OpenAI(api_key=key)

json_schema = {'type': 'object',
               'properties': {
                   'question': 'string',
                   'choices': {
                       'type':'array',
                       'items':{
                           'type':'string'
                       }
                   },
                   'answer':'string'
               },
               'required': ['question', 'choices', 'answer']
            }

json_schema_open = {'type': 'object',
               'properties': {
                   'question': 'string',
                   'answer':'string'
               },
               'required': ['question', 'answer']
            }


def generate_question_with_choices(document):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a teacher constructing an exam for a student based off a large amount of information. Given a particular chunk of that information, generate a question and 4 multiple choice answers of which 1 is correct. As you continue to generate choices and answers, make sure to give equal weighting to the possibility of the correct answer being A, B, C, or D. Focus on testing the details presented in the chunk."},
                {"role" : "user", "content":  
                f"Create a strictly formatted JSON object based on the provided text: '{document}'. The JSON schema is given for your reference: {json_schema}"
                "Ensure that the JSON object contains only the keys 'question', 'choices', and 'answer', "
                "where 'question' is a clear and concise query about the text, 'choices' is an array of four plausible answers, "
                "and 'answer' is one of the letters 'A', 'B', 'C', or 'D' corresponding to the first, second, third, or fourth option respectively in the 'choices' array."
            }],
            response_format={"type":"json_object"}
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")



def generate_open_ended_questions(document, path, code_base=True):

    # add path if dealing with a code base
    code_base_addition = ""
    if code_base:
        code_base_addition = f"This text can be found at this path in the larger code base '{path}'."

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a teacher constructing an exam for a student based off a large amount of information. Given a particular chunk of that information, generate a question and a correct answer. Focus on testing the details presented in the chunk. The questions will be provided separately from the text chunk."},
                {"role" : "user", "content":  
                f"Create a strictly formatted JSON object based on the provided text: '{document}'. {code_base_addition} The JSON schema is given for your reference: {json_schema_open}"
                "Ensure that the JSON object contains only the keys 'question', and 'answer', "
                "where 'question' is a clear and concise query about the text and 'answer' is the correct answer to that question"
            }],
            response_format={"type":"json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")


def append_questions_to_dataset(file_path, question_file, changed_dataset_file, open_ended):
    new_entries = []
    old_changed_entries = []
    count = 1
    with open(file_path, 'r+') as file:
        data = json.load(file)
        for entry in data:
            document = entry.get('document', '')
            path = entry.get('path', '')
            
            # entry['questions'] = []  # Initialize the questions list for each entry
            question_count = floor(entry.get('size', )/1000)+1
            #cap at 4 questions
            if question_count>4:
                question_count=4
            
            old_changed_entries.append(entry) 
            for _ in range(question_count):  # Assuming you want to generate 4 questions per entry
                print("\nQUESTION #", count,'\n')
                count+=1
                question_data_json_string = None

                if open_ended:
                    question_data_json_string = generate_open_ended_questions(document, path)
                else:
                    question_data_json_string = generate_question_with_choices(document)

                if question_data_json_string:
                    try:
                        question = json.loads(question_data_json_string)

                        # add the type if this is going into the training set
                        if open_ended:
                            question['type'] = 'qna'
                        print(question)
                        new_entries.append(question)  # Append the question to the new entry's questions list
                        old_changed_entries.append(question)

                    except json.JSONDecodeError as e:
                        print(f"Failed to decode JSON: {e}")
                        print("Issue with output: "+question_data_json_string)
                else:
                    print("None existent output: ", question_data_json_string)

             
            
        
    with open(question_file, 'w') as new_file:
        json.dump(new_entries, new_file, indent=4)
    
    with open(changed_dataset_file, 'w') as new_file:
        json.dump(old_changed_entries, new_file, indent=4)


dataset_path = "train/flamethrower_dataset.json"

changed_dataset_path = "train/train_and_eval_questions/flamethrower_dataset_with_questions_test.json"
questions = "eval_questions/flamethrower_questions_test.json"

changed_dataset_path_open = "train/train_and_open_questions/flamethrower_dataset_with_open_questions_test.json"
open_questions = "open_questions/flamethrower_questions_test.json"

append_questions_to_dataset(dataset_path, open_questions, changed_dataset_path_open, open_ended=True)
