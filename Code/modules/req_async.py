from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import asyncio
import httpx
import tqdm


async def get_values(client, url):
    try:
        resp = await client.get(url)
    except: # write error-urls in a file
        with open('error_urls.txt', 'a') as f:
            f.write(f'{url}\n')
        return None
    soup = BeautifulSoup(resp.content, 'html.parser')
    job_elements = soup.find(id='content').findAll('span', {"id" : re.compile('MainContentPlaceHolder*')})
    
    if job_elements[0].text == '':
        return None
    else:
        return [i.text for i in job_elements]


async def main(range_shift):
    '''
    Scrapes data from 10k NIST database entries
    in the range of range_shift*10_000 to (range_shift+1)*10_000
    '''
    async with httpx.AsyncClient(timeout=None) as client:

        tasks = []
        for number in range(range_shift*10_000, (range_shift+1)*10_000):
            url = f'https://srdata.nist.gov/xps/XPSDetailPage.aspx?AllDataNo={number}'
            tasks.append(asyncio.ensure_future(get_values(client, url)))

        values = await asyncio.gather(*tasks)
        all_values = [i for i in values if i is not None]
        return all_values

url = 'https://srdata.nist.gov/xps/XPSDetailPage.aspx?AllDataNo=10'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
job_elements = soup.find(id='content').findAll('span', {"id": re.compile('MainContentPlaceHolder*')})
keys = [i.get_attribute_list('id')[0][26:] for i in job_elements]  # keys

for i in tqdm.tqdm(range(30)): # read 300k urls
    vals = asyncio.run(main(i))
    df = pd.DataFrame(vals, columns=keys)
    if len(df) == 0:
        print(f'Not data for {i}k')
        continue
    else:
        df.to_parquet(f"db_full_{i}0k.prq")
