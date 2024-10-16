import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import json
from chromadb.config import Settings
import requests
import os
import ast # yl
from chromadb import Client

# os.system('git pull')

url = "https://jsearch.p.rapidapi.com/search"

querystring_ds = {"query":"Data scientist","page":"1","num_pages":"9","date_posted":"today","employment_types":"FULLTIME, CONTRACTOR","exclude_job_publishers":"Dice, jooble, Clearance Jobs, Geebo, Talent.com"}
querystring_ml = {"query":"Machine learning","page":"1","num_pages":"9","date_posted":"today","employment_types":"FULLTIME, CONTRACTOR","exclude_job_publishers":"Dice, jooble, Clearance Jobs, Geebo, Talent.com"}
querystring_ai = {"query":"AI","page":"1","num_pages":"9","date_posted":"today","employment_types":"FULLTIME, CONTRACTOR","exclude_job_publishers":"Dice, jooble, Clearance Jobs, Geebo, Talent.com"}

# "date_posted":"3days" "today"
api_key = os.getenv("RAPID_API_KEY")
headers = {
	"X-RapidAPI-Key": api_key,
	"X-RapidAPI-Host": "jsearch.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring_ds)
response_ml = requests.get(url, headers=headers, params=querystring_ml)
response_ai = requests.get(url, headers=headers, params=querystring_ai)

job_postings_json = response.json()


job_postings_json_ml = response_ml.json()
job_postings_json_ai = response_ai.json()

# This json file is created under the same path with vector_db
file_path = "data-ds.json"
with open(file_path, "w") as f:
    json.dump(job_postings_json['data'], f) # this generates the file data-ds.json
file_path = "data-ml.json"
with open(file_path, "w") as f:
    json.dump(job_postings_json_ml['data'], f)
file_path = "data-ai.json"
with open(file_path, "w") as f:
    json.dump(job_postings_json_ai['data'], f)

df_ds = pd.read_json('data-ds.json', encoding = 'utf-8')
df_ml = pd.read_json('data-ml.json', encoding = 'utf-8')
df_ai = pd.read_json('data-ai.json', encoding = 'utf-8')
df = pd.concat([df_ds, df_ml, df_ai], axis=0)

df_select = df[
      ['job_title', 'employer_name', 'employer_logo', 'employer_website',
       'employer_company_type', 'job_publisher', 'job_employment_type',
       'job_apply_link', 'job_description',
       'job_is_remote', 'job_city', 'job_state',
       'job_latitude', 'job_longitude', 'job_benefits',
       'job_required_experience', 'job_required_skills',
       'job_required_education', 'job_experience_in_place_of_education',
       'job_highlights']
].copy()

# yl: save job postings info before cleaning to check how to clean the data
from pathlib import Path

# yl: get the path of vector_db.py
srcpath = os.path.abspath(__file__)

# yl: get the folder where vector_db is located
srcdir = os.path.dirname(srcpath)

# yl: create the file path by combining the folder and the file name
filepath = Path(os.path.join(srcdir, 'before_clean.csv'))

# yl: save the df_select to before_clean.csv
df_select.to_csv(filepath) 


def convert_to_dict(val):
    if isinstance(val, dict):
        return val  # It's already a dictionary, so return as is
    elif isinstance(val, str):
        try:
            return ast.literal_eval(val)  # Convert string to dictionary
        except (ValueError, SyntaxError):
            return None  # Handle any errors gracefully
    return None  # In case of any unexpected type


def clean_job_postings(df):
  df['job_location'] = df['job_city'] + ', ' + df['job_state']
  df['info'] = df['job_title'] + '|' + df['job_location'] + '|' + df['employer_name']
  df = df.drop_duplicates(subset='info', ignore_index=True) # yl: reset index after remove duplicates
  df = df.drop_duplicates(subset='job_description', ignore_index=True) 
  df = df[~df['job_publisher'].str.contains('Geebo')] # filter out job postings from geebo.com
  df['job_required_experience'] = df['job_required_experience'].apply(convert_to_dict) # yl: convert string to dict
  df_exp = pd.json_normalize(df['job_required_experience']) # yl: extract key:value paires as multiple columns
  required_experience = df_exp['required_experience_in_months'] # yl: only keep 'required_experience_in_months'
  # df = pd.concat([df.drop(columns=['job_required_experience']), df_exp], axis=1)
  df = pd.concat([df, required_experience], axis=1) # yl: add'required_experience_in_months' as a new column to original df
  df['required_experience_in_months'] = df['required_experience_in_months'].fillna(0.0)
  df['required_experience'] = df['required_experience_in_months'].astype(int) / 12
  df['citizenship'] = df['job_description'].str.contains(r'clearance|SCI|US citizenship|US Citizen', case=False) # yl: simplify code

  return df

# don't forget to read_csv!
df_select = clean_job_postings(pd.read_csv(filepath)) 

# Set up the DataFrame
job_postings = df_select
job_postings = job_postings.dropna(subset=['info', 'job_description'])
job_postings = job_postings.fillna('') # na is not filled if use in clean_job_postings
job_postings.to_csv('jobs.csv', index=False)

# update vector database from DataFrame 
def update_chroma_db(df, collection):
  for index, row in df.iterrows():
    collection.add(
      # documents: associated with the vector. This could be used for text similarity search or other vector-based operations.
      documents=row['job_description'],
      # Metadata provides additional context or information about the vector
      metadatas=[{"info": row['info'],
                  "minimum": row['required_experience'],
                  "citizen": row['citizenship'],
                  "link": row['job_apply_link']
                 }],
      # the ids parameter is optional, but it's necessary if you need to uniquely identify, update, or retrieve specific vectors later
      ids=str(index)
    )
  return collection


# Get chromaDB client
setting = Settings(allow_reset = True)
chroma_client = chromadb.PersistentClient(path='.', settings = setting)

#chroma_client.delete_collection(name="job_postings")
#chroma_client.reset()
# collection = chroma_client.create_collection(
#         name="job_postings",
#         metadata={"hnsw:space": "cosine"}
#     )


collection = chroma_client.get_or_create_collection(
        name="job_postings",
        metadata={"hnsw:space": "cosine"}
    )

update_chroma_db(job_postings, collection)

# os.system(f'git add .')
# os.system(f"git commit -m 'update db'")
# os.system('git push')
# os.system('git add .')