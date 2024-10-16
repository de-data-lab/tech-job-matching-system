# __import__('pysqlite3')
import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# from dotenv import load_dotenv
import json
import os, time
from os import path, listdir
import numpy as np
import pandas as pd
import PyPDF2 as pdf
import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import chromadb
import cohere
from chromadb.utils import embedding_functions
from parse_resume import resume_parser

#--------------------------------------------LLM (Gemini pro) API-----------------------------------------------------------#
# load gemini pro LLM model API from environment variable
# load_dotenv() not needed as it is already stored environment variable in my system, use os.environ
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_gemini_response(input,pdf_content,prompt):
    # the higher temperature, more creative responses you will get
    generation_config = {
        "temperature": 0.0
    }
    # remove automated response blocking
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
    }
    model=genai.GenerativeModel('gemini-pro')

    results = ''
    try:
        if input:
        # The output is a single piece of generated text that takes into account all the input components provided in the list.
            response=model.generate_content([prompt,'job description:'+input,'resume:'+pdf_content], generation_config=generation_config)
        else:
            response=model.generate_content([prompt, 'resume:'+pdf_content], safety_settings=safety_settings)
        results = response.text
    # for some prompts Gemini will avoid generating the results even if you set all the filters to none.
    except ValueError:
        # If the response doesn't contain text, check if the prompt was blocked.
        st.write(response.prompt_feedback)
        # Also check the finish reason to see if the response was blocked, if so, will show 'FinishReason.SAFETY'
        st.write(response.candidates[0].finish_reason)
        # If the finish reason was SAFETY, the safety ratings have more details.
        st.write(response.candidates[0].safety_ratings)
        results = 'Encounter Error with Gemini AI Model'
    # except InternalServerError:
    #     st.write(response.prompt_feedback)
    #     results = 'Encounter Error with Gemini AI Model'
    except Exception as err:
        st.write(err)
        results = 'Encounter Error with Gemini AI Model'	    
    return results

# Generate prompts to generate resume revision and cover letter template
input_prompt_resume_summary = """
You are an skilled Applicant Tracking System scanner with a deep understanding of Applicant Tracking System functionality, please 
read the following resume carefully and summarize it within 500 word to include the following information in the resume step by step.
If there is a summary included, copy the summary directly to your response and append the following contents.
Please first find the important skills and tools included in the resume. 
Then conclude the background including all the work experience, do not need to include the specific number in each experience.
and projects in the resume. Finally summarize the education background with the highest degree level and the area of study and double 
check to omit the university or school attended. In other words, do not include the university attended.
"""

input_prompt_resume1 = """
You are a skilled Applicant Tracking System scanner with a deep understanding of Applicant Tracking System functionality, 
your task is to evaluate the resume against the provided job description. 
Find out the requirements that make this resume disqualified for this job in a list. Please first check the required or basic qualification,
then move on to the preferred qualifications.
Please limit the list no more than four most important bullet points and no more than 30 words for each bullet points.
"""

input_prompt_resume2 = """
You are submitting a resume to a job with the provided job description. 
Find out the requirements in the job description you should add to make you qualify for this job.
Please limit the list no more than three most important bullet points and no more than 30 words for each bullet points.
"""

input_prompt_cover_letter = """
You are the applicant who applied for this job and want to compose a strong but concise cover letter to convince the employer you have the skills and the experience for this job.
The first paragraph of the  cover letter must briefly discuss the your background, including both experience and projects. 
The second paragraph discuss how the applicant fit this role based on your skillsets matches the job requirements. Do not include the skillset not in the applicant's resume.
The third paragraph discuss the your interest in this role and thanks for the consideration.
Please limit the word count of cover letter no more than 300 words.
"""
#---------------------------------------------------Vector Database-------------------------------------------------------#

# Get vector database collection from local storage
chroma_client = chromadb.PersistentClient(path='.')
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_collection(name="job_postings")

# Find the most relevant job description and return the job posting information 
def get_relevant_ids(query, db, count=3, exclude_citizen_only = False, year_min = 0, year_max = 30):
    # yl: optimized logic based on 'exclude_citizen_only' (no filter at the beginning)
    where = {   
            # $and return results that match all of the filters in the list 
            "$and": [
        
            {
                "minimum": {
                    "$lte": year_max
                }
            },
            {
                "minimum": {
                    "$gte": year_min
                }
            }
        ]     
    }
    # yl: optimized logic based on 'exclude_citizen_only' (add filter if want to exlude citizen only jobs)
    if exclude_citizen_only:
        where["$and"].append(
                        {
                            # citizen is equal to FALSE
                            "citizen": {
                                "$eq": False
                            }
                        }
                      )    

    passage = db.query(query_texts=[query],
                     n_results=count, 
                     include = ["distances", "documents", "metadatas"],
                     # where filter is used to filter by metadata, must supply a filter dictionary
                     where=where
                     )
    # the outer list only contains one element (another list), so [0] effectively flattens this 2D structure into a 1D list
    ids = passage['ids'][0]
    cos = passage['distances'][0]
    doc = passage['documents'][0]
    metadata = passage['metadatas'][0]
    return ids, cos, doc, metadata

# query results example
# {
#   'documents': [[
#       'This is a document about pineapple',
#       'This is a document about oranges'
#   ]],
#   'ids': [['id1', 'id2']],
#   'distances': [[1.0404009819030762, 1.243080496788025]],
#   'uris': None,
#   'data': None,
#   'metadatas': [[None, None]],
#   'embeddings': None,
# }


# Upload resume
resume = ''
@st.cache_data
def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text
#---------------------------------------------------Rerank---------------------------------------------------------#
#
cohere_api_key=os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_api_key)
def rerank_results(co, query, docs, n = 3):
    results = co.rerank(model = 'rerank-english-v2.0', query = query, documents = docs, top_n = n)
    return results

#---------------------------------------------------Website---------------------------------------------------------#
# Page setup
st.title("Data Science Job Matching and Resume Enhancement")
st.markdown("Powered by Gemini Pro and Chroma vector database to help you find the most relevant \
         job openings and provide specific resume revision suggestion and cover letter template.")
st.markdown("Please be patient while waiting for the LLM-generated suggestions.") 
st.divider()

# Sidebar for user interaction
# change to bool return results as mentioned in official doc
submit = False
exclude_citizen_only = True
resume_summary = ''

with st.sidebar:
    # Using st.file_uploader does not save the file automatically
    # help here is a tooltit next to the upload button
    uploaded_file=st.file_uploader("Upload Your Resume",type="pdf",help="Please upload the pdf, the app won't save your resume")
  
    if uploaded_file is not None:
        st.write("PDF Uploaded Successfully")
        # get text format of uploaded resume
        resume = input_pdf_text(uploaded_file)
        # parsed resume and extracted info including summary, skills, experience and education as a single str
        resume_parsed = resume_parser(resume)
        resume_summary = get_gemini_response(input = None,pdf_content = resume_parsed,prompt = input_prompt_resume_summary)

    result_count = st.number_input('Results count', 1, 100, 30)
    # inserts a blank line or a small amount of vertical space in the Streamlit app (add a bit of spacing between widgets)
    st.write('')

    citizenship_included = st.checkbox('Include US citizen only job')
    if citizenship_included:
        exclude_citizen_only = False

    cohere_included = st.checkbox('Include Cohere reranking (More time needed)')

    year_min = st.slider('Minimum years of experience required', 0, 20, 0)
    year_max = st.slider('Maximum years of experience required', 0, 20, 20)

    submit = st.button("Generate LLM-powered results")
    if submit and resume_summary != '':
        st.markdown('## Resume Summary:')
        st.markdown(resume_summary)

# Show results 
# not inside sidebar
if submit and resume_summary != '':
    # Print summarized resume by LLM
    # Perform embedding search with vector database
    # return ids, cos, doc, metadata values (results=ids, scores=cos similarity or distance, doc=doc, meta=metadata)
    results, scores, doc, meta = get_relevant_ids(resume_summary, collection, result_count, exclude_citizen_only, year_min, year_max)
    if cohere_included:
        rerank_results = rerank_results(co, query = resume_summary, docs = doc, n = result_count)

    
    st.markdown('## Matched jobs')
    # All the elements inside this container are grouped together in the layout.
    with st.container():
        # results is a list of dictionaries like this
        # "results": [
        #     {
        #       "index": 3,
        #       "relevance_score": 0.999071
        #     },
        #     {
        #       "index": 4,
        #       "relevance_score": 0.7867867
        #     },
        #     {
        #       "index": 0,
        #       "relevance_score": 0.32713068
        #     }
        #   ]
        for index in range(len(results)):
            if cohere_included:
                i = rerank_results.results[index].index
                score = rerank_results.results[index].relevance_score
            else:
                i = index	
                score = 1 - scores[i]
		    # Insert a multi-element container that can be expanded/collapsed.
            # doc[i] is each job's description when generate vectors in chromadb
            with st.expander(meta[i]['info']):
                st.markdown(f'Similarity score: %.2f' %(score))
                st.markdown('**Job Description**')
                st.write(doc[i])
                st.link_button("Apply it!", meta[i]['link'], type="primary")

                response=get_gemini_response(doc[i],resume,input_prompt_resume1)
                st.subheader("Disqualifications")
                st.write(response) 
      #           try:
      #               st.write(response)
      #           except ValueError:
      #               # If the response doesn't contain text, check if the prompt was blocked.
      #               st.write(response.prompt_feedback)
		    # # Also check the finish reason to see if the response was blocked.
      #               st.write(response.candidates[0].finish_reason)
		    # # If the finish reason was SAFETY, the safety ratings have more details.
      #               st.write(response.candidates[0].safety_ratings)    

                response=get_gemini_response(doc[i],resume,input_prompt_resume2)
                st.subheader("Skills you may want to add")
                st.write(response)

                response=get_gemini_response(doc[i],resume,input_prompt_cover_letter)
                st.subheader("Coverletter")
                st.write(response)

                if i % 5 == 0:
                    time.sleep(5)
