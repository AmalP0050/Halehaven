import requests

short_url = ["XXXX t.co /namDL4YHYu",
 'XXXX t.co /MjvmV',
 'XXXX t.co /JSjtxfaxRJ',
 'XXXX t.co /xxGSANSE8K',
 'XXXX t.co /ZRhf5gWNQg']

for url in short_url:
    r = requests.get(url, allow_redirects=False)
    try:
        print(url, r.headers['location'])
    except KeyError:
        print(url, "Page doesn't exist!")