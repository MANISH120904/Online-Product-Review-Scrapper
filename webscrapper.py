import time
import random
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from selenium import webdriver
from wordcloud import WordCloud, STOPWORDS
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from nltk.sentiment import SentimentIntensityAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Download NLTK VADER lexicon
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Set up Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in the background
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Automatically manage ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL of Flipkart reviews page
url = "https://www.flipkart.com/hp-amd-ryzen-5-hexa-core-5500u-8-gb-512-gb-ssd-windows-11-home-15s-eq2144au-thin-light-laptop/product-reviews/itmd57b41ed8750a?pid=COMGBH9JDPVGD8BH&lid=LSTCOMGBH9JDPVGD8BHIZPQIL&marketplace=FLIPKART"
driver.get(url)
time.sleep(5)  # Allow page to load

# Scroll and load more reviews
review_class = "ZmyHeo"  # Flipkart's review class
all_reviews = set()  # Store unique reviews
max_pages = 10  # Maximum number of pages to scrape

for page in range(max_pages):
    try:
        # Wait for reviews to load
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, review_class)))

        # Extract reviews
        reviews = driver.find_elements(By.CLASS_NAME, review_class)
        for review in reviews:
            text = review.text.strip()
            if text:
                all_reviews.add(text)

        logging.info(f"Collected {len(all_reviews)} reviews so far...")

        # Click the "Next" button to load more reviews
        next_buttons = driver.find_elements(By.XPATH, "//span[text()='Next']")
        if next_buttons:
            driver.execute_script("arguments[0].click();", next_buttons[0])
            time.sleep(3)  # Wait for the next page to load
        else:
            break  # No more pages

    except Exception as e:
        logging.warning(f"Error encountered: {e}")
        break  # Stop if no more pages or errors occur

# Close the browser
driver.quit()

# Convert reviews to DataFrame
df = pd.DataFrame(list(all_reviews), columns=["Review"])

# Check if reviews were extracted
if df.empty:
    logging.error("No reviews found! The website structure may have changed.")
    exit()

logging.info(f"Total {len(df)} Reviews Scraped Successfully!")

# Select 50 random reviews
if len(df) > 50:
    df = df.sample(n=50, random_state=42).reset_index(drop=True)

logging.info(f"Randomly selected 50 reviews.")

# Sentiment Analysis Function
def get_sentiment(text):
    sentiment = sia.polarity_scores(text)
    polarity = "Positive" if sentiment["compound"] > 0.05 else "Negative" if sentiment["compound"] < -0.05 else "Neutral"
    return sentiment["compound"], polarity

# Apply sentiment analysis
df["Sentiment_Score"], df["Polarity"] = zip(*df["Review"].map(get_sentiment))

# Assign Star Ratings based on sentiment score
def assign_star_rating(score):
    if score >= 0.6:
        return 5
    elif score >= 0.3:
        return 4
    elif score >= 0.0:
        return 3
    elif score >= -0.3:
        return 2
    else:
        return 1

df["Star_Rating"] = df["Sentiment_Score"].apply(assign_star_rating)

# Display Results
logging.info("\nExtracted Reviews & Sentiment Analysis:")
logging.info(df.head())

# Save as CSV
df.to_csv("random_reviews.csv", index=False)
logging.info("Data saved as 'random_reviews.csv'")

# Visualization - Sentiment Distribution
sentiment_counts = df["Polarity"].value_counts()

plt.figure(figsize=(7, 7))
colors = ["green", "gray", "red"]
plt.pie(
    sentiment_counts, 
    labels=sentiment_counts.index, 
    autopct="%1.1f%%", 
    colors=colors, 
    startangle=140,
    explode=[0.05] * len(sentiment_counts)
)
plt.title("Sentiment Distribution of Reviews")
plt.show()

# Star Rating Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=df["Star_Rating"], palette="viridis")
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.title("Star Rating Distribution")
plt.xticks(rotation=0)
plt.show()

logging.info("Visualization completed successfully!")

# Generate WordCloud
text = " ".join(review for review in df["Review"])

# Define stopwords (remove common & product-specific words)
stopwords = set(STOPWORDS)
stopwords.update(["product", "laptop", "hp", "device"])  # Customize as needed

wordcloud = WordCloud(
    width=800, height=400,
    background_color="black",
    colormap="coolwarm",
    stopwords=stopwords,
    max_words=100
).generate(text)

# Plot WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequently Used Words in Reviews", fontsize=14)
plt.show()

logging.info("WordCloud visualization completed!")
