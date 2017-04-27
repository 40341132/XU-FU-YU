from collections import Counter
import csv


def get_domain(email_address):
    """split on '@' and return the last piece"""
    return email_address.lower().split("@")[-1]
with open('email.txt', 'r') as f:
    domain_counts = Counter(get_domain(line.strip())
    for line in f
        if "@" in line)
print(domain_counts)


def process(date, symbol, price):
    print date, symbol, price

with open('tab_delimited_stock_prices.txt', 'rb') as f:
 reader = csv.reader(f, delimiter='\t')
 for row in reader:
    date = row[0]
    symbol = row[1]
    closing_price = float(row[2])
    process(date, symbol, closing_price)

with open('colon_delimited_stock_prices.txt', 'rb') as f:
 reader = csv.DictReader(f, delimiter=':')
 for row in reader:
    date = row["date"]
    symbol = row["symbol"]
    closing_price = float(row["closing_price"])
    process(date, symbol, closing_price)


