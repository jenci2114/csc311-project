import pandas as pd
import numpy as np


def process_student_meta(filepath: str) -> dict:
    """
    Process the student metadata
    Output is dictionary of dictionary where
    dict[i] = {'age': age of student with user_id i, 'gender': (resp.), 'premium': (resp.)}
    """
    student_meta = pd.read_csv(filepath)
    # Student meta data processing
    student_user_id = student_meta['user_id'].values  # user id array

    student_birthdate = student_meta['data_of_birth'].values  # birthdate array
    existing_ages = []
    for i in range(len(student_birthdate)):
        if not pd.isnull(student_birthdate[i]):
            existing_ages.append(2022 - int(student_birthdate[i][:4]))
    mean_age = np.mean(existing_ages)
    student_age = []  # age array
    for i in range(len(student_birthdate)):
        if pd.isnull(student_birthdate[i]):
            student_age.append(mean_age)
        else:
            student_age.append(2022 - int(student_birthdate[i][:4]))

    student_gender = student_meta['gender'].values  # gender array

    student_premium = student_meta['premium_pupil'].values  # premium array
    existing_premiums = []
    for i in range(len(student_premium)):
        if not pd.isnull(student_premium[i]):
            existing_premiums.append(student_premium[i])
    mean_premium = np.mean(existing_premiums)
    student_premium_cleaned = []  # premium array
    for i in range(len(student_premium)):
        if pd.isnull(student_premium[i]):
            student_premium_cleaned.append(mean_premium)
        else:
            student_premium_cleaned.append(student_premium[i])

    student_meta_dict = {}
    for i in range(len(student_user_id)):
        student_meta_dict[student_user_id[i]] = {
            'age': student_age[i],
            'gender': student_gender[i],
            'premium': student_premium_cleaned[i]
        }

    return student_meta_dict


def process_question_meta(filepath: str, num_questions: int, num_subjects: int) -> np.array:
    """
    Process the question metadata
    Output is a numpy array of shape (num_questions, num_subjects)
    where arr[i, j] is 1 if question i is of subject j and 0 otherwise
    """
    question_meta = pd.read_csv(filepath).to_numpy()
    meta_mat = np.zeros((num_questions, num_subjects))
    for question, subject_str in question_meta:
        subject_list = subject_str[1:-1].split(',')
        subject_list = [int(x) for x in subject_list]
        for subject in subject_list:
            meta_mat[question, subject] = 1
    return meta_mat


def get_subject_number(filepath: str) -> int:
    """Obtain the number of subjects from subject metadata, whose filepath is as given."""
    subject_meta = pd.read_csv(filepath)
    return max(subject_meta['subject_id']) + 1
