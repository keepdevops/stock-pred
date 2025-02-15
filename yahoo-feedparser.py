import feedparser

# Construct the RSS feed URL for the stock
rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={aapl}®ion=US&lang=en-US"

# Parse the RSS feed
news_feed = feedparser.parse(rss_url)

# Print the latest news items
for entry in news_feed.entries[:5]:  # Limit to 5 for demonstration
    print(f"Title: {entry.title}")
    print(f"Published: {entry.published}")
    print(f"Link: {entry.link}")
    print("\n")
