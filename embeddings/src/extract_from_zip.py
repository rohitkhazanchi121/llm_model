import zipfile
import os
from bs4 import BeautifulSoup
import pandas as pd
def clean_html(text):
    """Remove HTML tags using BeautifulSoup"""
    return BeautifulSoup(text, "html.parser").get_text()

def get_url_of_page(row):
    return f'https://teckresources.atlassian.net/wiki/spaces/DASA/pages/{row["contentid"]}/{row["title"]}'.replace(' ', '+')

def extract_from_zip(zip_path, extract_path):

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        print('Extracted to', extract_path)

    text_files = [os.path.join(extract_path, file) for file in os.listdir(extract_path) if file.startswith('bodycontent')]
    content_files = [os.path.join(extract_path, file) for file in os.listdir(extract_path) if file.startswith('content.csv')]
    
    for file in content_files:
        contents_data = pd.read_csv(file, compression=None)
        contents_data['url'] = contents_data.apply(get_url_of_page, axis=1)
    
    for file in text_files: 
        body_data = pd.read_csv(file, compression=None)
        body_data = body_data.dropna(subset=['body'])
        body_data['body'] = body_data['body'].apply(clean_html)

    combined_data = pd.merge(body_data, contents_data[['contentid', 'url']], on='contentid')
    print('Documents:', combined_data.head(3))

    return combined_data    

zip_path = '/Users/rohit.khazanchi@teck.com/repo/da-hackathon-chatbot/embeddings/data/Confluence-export-teckresources.atlassian.net-DASA-csv.zip'
extract_path = '/Users/rohit.khazanchi@teck.com/repo/da-hackathon-chatbot/embeddings/data/extracted/DASA-csv'


extracted_text = extract_from_zip(zip_path,extract_path )
