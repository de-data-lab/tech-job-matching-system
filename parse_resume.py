import pandas as pd
import numpy as np
import spacy

RESUME_SECTIONS = [
    'summary',
    'education',
    'experience',
    'skills',
    'projects',
    'certifications',
    'licenses',
    'awards',
    'honors',
    'publications',
    'references',
]

def resume_parser(resume):
    """
    Remove the stop words and punctuation.
    Extract summary, skills, experience and education part from the resume.

    Args:
        resume (str): The string read from pdf.

    Returns:
        str: A string containing all the extracted information.
    """

    # Load the English model
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(resume)
    doc = clean_text(doc)


    summary = extract_summary(doc)
    skills = extract_skills(doc)
    experience = extract_experience(doc)
    education = extract_education(doc)

    result = summary + '\n' + skills + '\n' + experience + '\n' + education

    return result.lower()

def clean_text(tokens):
    tokens = [token for token in tokens if (not token.is_punct and not token.is_space and not token.is_stop)]
    return tokens

def extract_summary(doc):
    """
    Extract summary from a given string. It does so by using the Spacy module.

    Args:
        text (str): The string from which to extract summary.

    Returns:
        str: A string containing all the extracted summary.
    """
    summary_section = []
    in_summary_section = False

    for token in doc:
        # print(token)
        if token.text.lower() in RESUME_SECTIONS and token.text not in RESUME_SECTIONS:
            if token.text in ['Summary', 'SUMMARY']:
                in_summary_section = True
            else:
                in_summary_section = False

        if in_summary_section:
            summary_section.append(token.text)

    return ' '.join(summary_section)

def extract_skills(doc):
    """
    Extract skills from a given string. It does so by using the Spacy module.

    Args:
        text (str): The string from which to extract skills.

    Returns:
        str: A string containing all the extracted skills.
    """
    skills_section = []
    in_skills_section = False

    for token in doc:
        # print(token)
        if token.text.lower() in RESUME_SECTIONS and token.text not in RESUME_SECTIONS:
            if token.text in ['Skills', 'SKILLS', 'Skill', 'SKILL']:
                in_skills_section = True
            else:
                in_skills_section = False

        if in_skills_section:
            skills_section.append(token.text)

    return ' '.join(skills_section)


def extract_experience(doc):
    """
    Extract experience from a given string. It does so by using the Spacy module.

    Args:
        text (str): The string from which to extract experience.

    Returns:
        str: A string containing all the extracted experience.
    """
    experience_section = []
    in_experience_section = False

    for token in doc:
        # print(token)
        if token.text.lower() in RESUME_SECTIONS and token.text not in RESUME_SECTIONS:
            if token.text in ['Experience', 'EXPERIENCE', 'Projects', 'PROJECTS']:
                in_experience_section = True
            else:
                in_experience_section = False

        if in_experience_section:
            experience_section.append(token.text)

    return ' '.join(experience_section)


def extract_education(doc):
    """
    Extract nouns and proper nouns from education.

    Args:
        text (str): The input text to extract nouns from education.

    Returns:
        list: A list of extracted nouns.
    """
    pos_tags = ['NOUN', 'PROPN']
    education_section = []
    in_education_section = False

    for token in doc:
        # print(token)
        if token.text.lower() in RESUME_SECTIONS and token.text not in RESUME_SECTIONS:
            if token.text in ['Education', 'EDUCATION']:
                in_education_section = True
            else:
                in_education_section = False

        if in_education_section:
            if token.pos_ in pos_tags:
                education_section.append(token.text)

    return ' '.join(education_section)