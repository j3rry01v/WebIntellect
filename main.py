from googlesearch import search
import pandas as pd
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import pandas as pd
import spacy
import re
import pytextrank
import openai
import time
import os
import argparse
from dotenv import load_dotenv

# nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe("textrank")


def get_excel_file_name(csv_file_path):
    """
    Generates an Excel file name based on the input CSV file name.
    For example, if the input CSV file name is '/path/to/Moonsoft_new.csv',
    the output Excel file name will be 'analysis_of_Moonsoft_new.xlsx'.
    """
    base_name = os.path.basename(csv_file_path) 
    name_without_ext = os.path.splitext(base_name)[0]  
    excel_file_name = f"analysis_of_{name_without_ext}.xlsx" 
    return excel_file_name


def initialize_excel_file(excel_path, columns):
    """
    Initializes the Excel file with headers if it doesn't exist.
    """
    if not os.path.exists(excel_path):
        df = pd.DataFrame(columns=columns)
        df.to_excel(excel_path, index=False)


def query_gpt_with_keywords(cleaned_keyphrases,url, excel_path,attempt=1):
    """
    Generate a GPT prompt with the cleaned keyphrases and query GPT for the analysis.
    """
    keywords_str = ', '.join(cleaned_keyphrases)
    config_path = 'config.env'
    load_dotenv(dotenv_path=config_path)
    openai.api_key = os.getenv('OPEN_AI_API_KEY')
    if not openai.api_key:
        print("OPEN AI API key not found")
        return

    prompt_text = (f"Provide a detailed analysis of this keywords, focusing on its critical role in IT infrastructure "
                   f"observability and monitoring. Evaluate the impact of integrating artificial intelligence into IT "
                   f"observability solutions, emphasizing the advancements in operational efficiency and reliability for "
                   f"modern digital enterprises, give output as which domain or area of work this keywords are related to, "
                   f"output should be a single or multiple terms only no need of displaying \"Based on the detailed analysis "
                   f"and focus requested, the keywords you've provided relate to the following domains or areas of work: "
                   f"keywords are\n\n{keywords_str}")
    try:   
        conversation = [
        {"role": "system", "content": "You are a knowledgeable assistant who provides detailed analysis on keywords related to IT infrastructure observability and monitoring, including the integration of artificial intelligence."},
        {"role": "user", "content": prompt_text}
        ]
    
    # conversation = [
    #     {"role": "system", "content": "You are a knowledgeable assistant who analyzes keywords and determines their related domain or area of work, focusing on IT infrastructure observability and monitoring and the integration of artificial intelligence."},
    #     {"role": "user", "content": f"Analyze these keywords and determine the domain or area of work they are related to: {keywords_str}."}
    # ]
    
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
        messages=conversation
         )
        
    except openai.error.RateLimitError as e:
        wait_time = 20  # Wait time in seconds; adjust as needed based on the specific rate limit error message
        print(f"Rate limit reached, waiting for {wait_time} seconds before retrying...")
        time.sleep(wait_time)
        
        # Retry the request if the attempt count is below a certain threshold (e.g., 3 attempts)
        if attempt <= 3:
            print(f"Retrying attempt {attempt + 1}...")
            query_gpt_with_keywords(cleaned_keyphrases, attempt + 1)
        else:
            print("Max retry attempts reached, moving to the next task.")
    
    # response = openai.Completion.create(
    #     engine="gpt-3.5-turbo",  # or another version, depending on availability and needs
    #    prompt=prompt,
    #     temperature=0.5,
    #     max_tokens=256,
    #     top_p=1.0,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0
    # )
    response_text = response['choices'][0]['message']['content']
    # analysis_result = response.choices[0].text.strip()
    print(f"Analysis result for the keywords: {response_text}")
    lines = response_text.strip().split('\n')
    analysis_result = lines[0].replace('Analysis result for the keywords: Keywords:', '').strip()
    domain_area = lines[1].replace('Domain/Area:', '').strip() if len(lines) > 1 else ''

    df = pd.DataFrame([[url, analysis_result, domain_area]], columns=['URL', 'Analysis Result', 'Domain/Area'])
    with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, index=False, header=not os.path.exists(excel_path), startrow=writer.sheets['Sheet1'].max_row if 'Sheet1' in writer.sheets else 0)



def clean_phrase(phrase):
    """
    Cleans the phrase by removing numbers, dots, and commas.
    """
    return re.sub(r'[0-9.,]', '', phrase).strip()

def get_site_name_from_url(url):
    """
    Extracts the site name from a URL to use as a filename for the CSV.
    """
    pattern = re.compile(r'https?://(www\.)?')
    site_name = pattern.sub('', url).split('/')[0]
    return site_name.replace('.', '_')



def get_links_from_google(term, num_results=10, lang='en'):
    url_list = [x for x in search(term=term, lang=lang, num_results=num_results)]
    return pd.DataFrame(url_list, columns=['url'])



def process_urls_from_csv(input_csv_path,excel_path,limit):
    """
    Reads URLs from a CSV file, processes each URL to extract top keyphrases,
    prints the cleaned keyphrases (keywords) in the terminal,
    sends these keyphrases to GPT for analysis, and saves the results.
    """
    # 
    urls_df = pd.read_csv(input_csv_path)
    limited_urls_df = urls_df.head(limit)

    for url in limited_urls_df.iloc[:, 0]:
        try:
            site_name = get_site_name_from_url(url)
            text = get_page_content(url)
            text = text_cleaning(text)
            results_from_page = get_top_n_keyphrases(text, top_n=15)

            cleaned_keyphrases = [clean_phrase(phrase) for phrase in results_from_page.index]
            print(f"Cleaned keyphrases for {site_name}: {cleaned_keyphrases}")

            query_gpt_with_keywords(cleaned_keyphrases, url, excel_path)
        except Exception as e:
            print(f"Failed to process {url}: {e}")

def get_page_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
    }

    with HTMLSession() as session:
        try:
            res = session.get(url, headers=headers, timeout=200)
            return BeautifulSoup(res.content, 'html.parser').text
        except:
            return BeautifulSoup('', 'html.parser').text
        finally:
            print(f"Done Scrapping")


# Step 3. Cleaning the text content from pages

def text_cleaning(text):
    try:
        # removing more than one newline or spaces
        text = re.sub(r'[\n\r]+', '\n', text)
    except:
        print(f"Failed to clean")
    return text


# Step 4. Extracting Keyphrases from content
def get_top_n_keyphrases(text, top_n=25):
    # load a spaCy model, depending on language, scale, etc.
    nlp = spacy.load("en_core_web_sm")

    # add PyTextRank to the spaCy pipeline
    nlp.add_pipe("textrank")
    doc = nlp(text)

    # examine the top-ranked phrases in the document
    if top_n > len(doc._.phrases):
        top_n = len(doc._.phrases)

    rank_dict = {phrase.text: phrase.rank for phrase in doc._.phrases[:top_n]}
    return pd.DataFrame.from_dict(rank_dict, orient='index')


if __name__ == "__main__":
    # Get Top Key phrases of a particular topic
    # topic = 'Latest Mobiles In India'
    # df = get_links_from_google(topic, num_results=5, lang='en')
    # df['text'] = df['url'].apply(get_page_content)
    # df['text'] = df['text'].apply(text_cleaning)
    # results = get_top_n_keyphrases(''.join(df['text'].to_list()))
    # results.to_csv(f'top_key_phrase_for_{topic}.csv', encoding='utf-8-sig')
    # print('done')

    # Get top Key phrases from competitor;s page
    # url = 'https://www.entechreview.com/category/news/page/182/'
    # text = get_page_content(url)
    # text = text_cleaning(text)
    # results_from_page = get_top_n_keyphrases(text, top_n=15)
    # results_from_page.to_csv(f'top_key_phrase_for_competitors_page.csv', encoding='utf-8-sig')
    parser = argparse.ArgumentParser(description="Process URLs from a CSV file for GPT analysis.")
    parser.add_argument("--backlinks", type=str, required=True, help="Path to the input CSV file with URLs.")
    parser.add_argument("--limit", type=int, default=100, help="Limit on the number of URLs to process.")
    # input_csv_path = "/Users/jerin.sr/Downloads/Moongsoft.csv"
    # process_urls_from_csv(input_csv_path)
    args = parser.parse_args()

    # excel_path = 'gpt_analysis.xlsx'
    excel_path = get_excel_file_name(args.backlinks)
    columns = ['URL', 'Analysis Result', 'Domain/Area']
    initialize_excel_file(excel_path, columns)
    process_urls_from_csv(args.backlinks, excel_path, args.limit)
    # process_urls_from_csv(input_csv_path, excel_path)

    print('done')
