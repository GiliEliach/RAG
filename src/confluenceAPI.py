import os

from atlassian import Confluence
import requests
from config import CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_TOKEN, DATA_FILES


def get_pages_info(confluence_url):
    url = f"{confluence_url}/rest/api/content?spaceKey=Verisoft&type=page&limit=100&expand=version,space,title"
    pages = {}
    auth = (CONFLUENCE_USERNAME, CONFLUENCE_TOKEN)
    while url:
        # Perform the GET request
        response = requests.get(url, auth=auth)
        response.raise_for_status()  # This will raise an error for non-200 responses
        data = response.json()
        for result in data.get('results', []):
            pages[result['id']] = result['title']
            url = data.get('_links', {}).get('next')
    return pages


if __name__ == '__main__':
    if not os.path.exists(DATA_FILES):
        os.makedirs(DATA_FILES)
    pages_info = get_pages_info(CONFLUENCE_URL)
    confluence = Confluence(
            url=CONFLUENCE_URL,
            username=CONFLUENCE_USERNAME,
            password=CONFLUENCE_TOKEN,
            api_version='cloud'
        )

    for idx, (page_id, page_title) in enumerate(list(pages_info.items())):
        content = confluence.export_page(page_id)
        with open(DATA_FILES+f'/{page_title}.pdf', "wb") as pdf_file:
            pdf_file.write(content)
            print(f"Document {idx+1} of {len(pages_info.values())} is saved in the 'src/confluence' folder, the name "
                  f"of the document: "+page_title)

