import requests

def get_job_links(api_key, page=0, limit=25):
    """
    Fetches job postings from TheirStack API and returns a list of job URLs.
    Minimal required fields only.
    """
    url = "https://api.theirstack.com/v1/jobs/search"
    
    # Minimal required payload
    payload = {
        "page": page,
        "limit": limit,
        "job_country_code_or": ["US"],  # required by API
        "posted_at_max_age_days": 365   # any number, to get all recent jobs
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    data = response.json()
    
    jobs_list = data.get('jobs') or data.get('results') or []
    job_links = [job.get('url') for job in jobs_list if job.get('url')]
    
    return job_links

api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqYXlhbmRoYW5zNDg0QGdtYWlsLmNvbSIsInBlcm1pc3Npb25zIjoidXNlciIsImNyZWF0ZWRfYXQiOiIyMDI1LTA5LTI5VDA4OjA1OjUxLjEwNjU5MCswMDowMCJ9.a-DnheVoZDl2w9_3k30goHGWPf5pEb-tD_ipcDRRhrU" 
links = get_job_links(api_key)
print(links)
