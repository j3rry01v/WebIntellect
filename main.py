from googlesearch import search
import pandas as pd
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import pandas as pd
import spacy
import re


def process_urls_from_csv(input_csv_path):
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

process_urls_from_csv(input_csv_path=path)
print('done')
