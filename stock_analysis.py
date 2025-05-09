import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Use in-memory storage (temporary)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# Create or load a ChromaDB collection
collection = chroma_client.get_or_create_collection(name="stock_data", embedding_function=embedding_fn)

# Initialize Agents
def initialize_agents():
    # Web Search Agent
    web_search_agent = Agent(
        name="Web Search Agent",
        role="Search the web for information",
        model=Gemini(id="gemini-1.5-flash"),
        tools=[DuckDuckGo()],
        instructions=["Always include sources"],
        show_tool_calls=True,
        markdown=True,
    )

    # Financial Agent
    finance_agent = Agent(
        name="Finance AI Agent",
        model=Gemini(id="gemini-1.5-flash"),
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                company_news=True,
            ),
        ],
        instructions=["Use tables to display the data"],
        show_tool_calls=True,
        markdown=True,
    )

    # Multi-AI Agent
    multi_ai_agent = Agent(
        team=[web_search_agent, finance_agent],
        model=Gemini(id="gemini-1.5-flash"),
        instructions=["Always include sources", "Use tables to display the data"],
        show_tool_calls=True,
        markdown=True,
    )

    return web_search_agent, finance_agent, multi_ai_agent

# Function to parse raw data
def parse_raw_data(raw_data):
    """
    Parses raw data returned by the agent.
    """
    try:
        # Try parsing as JSON
        parsed_data = json.loads(raw_data)
        return parsed_data
    except json.JSONDecodeError:
        # If JSON parsing fails, inspect the raw data format
        st.warning("Raw data is not in JSON format. Inspecting data structure...")
        st.write("Raw Data:", raw_data)
        return None

def create_chat_interface():
    st.sidebar.subheader("Chat Interface")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the stock..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the query and get response
        with st.chat_message("assistant"):
            response = process_stock_query(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)

def process_stock_query(query):
    """
    Process user query about stock data using stored ChromaDB data.
    """
    try:
        # Query ChromaDB for relevant stock data
        results = query_chromadb(query)
        if not results:
            return "I'm sorry, I couldn't find any stored data about this stock. Please run an analysis first."

        # Parse the stored JSON data
        stored_data = json.loads(results[0])  # Assuming results returns a list with one item

        # Create a prompt for the Gemini model
        analysis_prompt = f"""
        Based on the following stored stock data and the user's question: "{query}"

        Analyst Recommendations: {stored_data.get('analyst_recommendations', 'Not available')}
        Recent News: {stored_data.get('news', 'Not available')}
        Financial Metrics: {stored_data.get('financial_metrics', 'Not available')}
        Future Outlook: {stored_data.get('future_outlook', 'Not available')}

        Please provide a detailed and specific answer to the user's question using this information.
        Focus on relevant data points and provide concrete examples where possible.
        If the question asks about something not covered in the data, please acknowledge that limitation.
        """

        # Initialize chat agent
        chat_agent = Agent(
            name="Stock Chat Agent",
            model=Gemini(id="gemini-1.5-flash"),
            instructions=[
                "Provide concise but informative answers",
                "Use bullet points for multiple data points",
                "Include specific numbers and metrics when available",
                "Acknowledge when information is not available or outdated"
            ],
            markdown=True
        )

        # Generate response
        response = chat_agent.run(analysis_prompt)
        return response.messages[-1].content

    except Exception as e:
        return f"I encountered an error while processing your query: {str(e)}"
# Function to analyze analyst recommendations
def analyze_analyst_recommendations(recommendations):
    """
    Analyzes analyst recommendations and returns insights.
    """
    insights = []
    for period, data in recommendations.items():
        strong_buy = data.get("strongBuy", 0)
        buy = data.get("buy", 0)
        hold = data.get("hold", 0)
        sell = data.get("sell", 0)
        strong_sell = data.get("strongSell", 0)

        # Calculate total ratings
        total_ratings = strong_buy + buy + hold + sell + strong_sell

        # Calculate percentages
        strong_buy_pct = (strong_buy / total_ratings) * 100
        buy_pct = (buy / total_ratings) * 100
        hold_pct = (hold / total_ratings) * 100
        sell_pct = (sell / total_ratings) * 100
        strong_sell_pct = (strong_sell / total_ratings) * 100

        # Generate insights
        insight = {
            "period": period,
            "strong_buy": strong_buy,
            "buy": buy,
            "hold": hold,
            "sell": sell,
            "strong_sell": strong_sell,
            "strong_buy_pct": strong_buy_pct,
            "buy_pct": buy_pct,
            "hold_pct": hold_pct,
            "sell_pct": sell_pct,
            "strong_sell_pct": strong_sell_pct,
        }
        insights.append(insight)

    return insights


@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

# Function to analyze news sentiment using FinBERT
def analyze_news_sentiment(news_texts):
    tokenizer, model = load_finbert()
    sentiments = []
    
    for text in news_texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get sentiment label and score
        sentiment_score = predictions.detach().numpy()[0]
        sentiment_label = ["negative", "neutral", "positive"][np.argmax(sentiment_score)]
        confidence = np.max(sentiment_score)
        
        sentiments.append({
            "text": text,
            "sentiment": sentiment_label,
            "confidence": float(confidence),
            "scores": {
                "negative": float(sentiment_score[0]),
                "neutral": float(sentiment_score[1]),
                "positive": float(sentiment_score[2])
            }
        })
    
    return sentiments

def analyze_company_news(news):
    """
    Analyzes company news and returns key highlights with sentiment analysis.
    """
    highlights = []
    news_texts = []
    
    for article in news:
        title = article.get("title", "")
        description = article.get("description", "")
        source = article.get("source", {}).get("name", "")
        url = article.get("url", "")
        
        # Combine title and description for sentiment analysis
        news_text = f"{title}. {description}"
        news_texts.append(news_text)
        
        highlight = {
            "title": title,
            "description": description,
            "source": source,
            "url": url,
        }
        highlights.append(highlight)
    
    # Get sentiment analysis for all news items
    sentiments = analyze_news_sentiment(news_texts)
    
    # Combine highlights with sentiment analysis
    for highlight, sentiment in zip(highlights, sentiments):
        highlight["sentiment"] = sentiment
    
    return highlights

# Function to fetch balance sheet
def fetch_balance_sheet(symbol):
    """
    Fetches the balance sheet for a given stock symbol.
    """
    ticker = yf.Ticker(symbol)
    balance_sheet = ticker.balance_sheet
    return balance_sheet

# Function to fetch P/E ratio and other financial metrics
def fetch_financial_metrics(symbol):
    """
    Fetches financial metrics like P/E ratio, market cap, etc.
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info
    metrics = {
        "P/E Ratio": info.get("trailingPE", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "Revenue Growth": info.get("revenueGrowth", "N/A"),
        "Profit Margins": info.get("profitMargins", "N/A"),
        "Dividend Yield": info.get("dividendYield", "N/A"),
    }
    return metrics

# Function to fetch past events (e.g., earnings, stock splits)
def fetch_past_events(symbol):
    """
    Fetches past events like earnings reports and stock splits.
    """
    ticker = yf.Ticker(symbol)
    events = {
        "Earnings Dates": ticker.earnings_dates,
        "Stock Splits": ticker.splits,
    }
    return events

# Function to generate future outlook
def generate_future_outlook(symbol, news, balance_sheet, financial_metrics, analyst_recommendations):
    """
    Generates a future outlook for the stock using the Gemini model via an Agent.
    """
    # Prepare the input prompt for the Gemini model
    print(symbol,news,balance_sheet)
    prompt = f"""
    Analyze the following data for the stock {symbol} and provide a future outlook:

    1. **Latest News**: {news}
    2. **Balance Sheet**: {balance_sheet.to_string()}
    3. **Financial Metrics**: {financial_metrics}
    4. **Analyst Recommendations**: {analyst_recommendations}

    Based on this data, provide a detailed future outlook for the stock, including potential growth, risks, and any other relevant insights.
    """

    # Initialize an Agent with the Gemini model
    outlook_agent = Agent(
        name="Future Outlook Agent",
        model=Gemini(id="gemini-1.5-flash"),
        instructions=["Provide a detailed and professional future outlook for the stock based on the given data."],
        markdown=True,
    )

    # Use the Agent to generate the future outlook
    response = outlook_agent.run(prompt)
    return response.messages[-1].content  # Extract the generated content

# Function to store data in ChromaDB
def serialize_pandas_data(obj):
    """
    Custom JSON serializer to handle Pandas Timestamp objects and DataFrames.
    """
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Modify the store_data_in_chromadb function
def store_data_in_chromadb(symbol, data):
    """
    Stores stock data in ChromaDB after converting unsupported data types.
    """
    # Convert data to a JSON string
    data_str = json.dumps(data, default=str)  # Use default=str to handle Timestamp and other unsupported types
    collection.add(
        documents=[data_str],  # Store the JSON string as a document
        metadatas=[{"symbol": symbol}],  # Metadata must use supported types
        ids=[symbol]  # ID must be a string
    )

# Function to query ChromaDB
def query_chromadb(query):
    """
    Queries ChromaDB for relevant information.
    """
    results = collection.query(
        query_texts=[query],
        n_results=1  # Retrieve the top result
    )
    if results["documents"]:
        return results["documents"][0]
    return None

# Main function
def main():
    st.title("Stock Analysis Application")
    
    # Create tabs for analysis and chat
    tab1, tab2 = st.tabs(["Stock Analysis", "Chat"])
    
    with tab1:
        # Move sidebar elements inside the tab to avoid duplication
        symbol = st.sidebar.text_input(
            "Enter Stock Symbol (e.g., NVDA):", 
            "NVDA",
            key="stock_symbol_input"  # Added unique key
        )
        
        if st.sidebar.button("Analyze", key="analyze_button"):  # Added unique key
            st.write(f"Analyzing {symbol}...")

            web_search_agent, finance_agent, multi_ai_agent = initialize_agents()

            # Input stock symbol
           

            # Fetch analyst recommendations
            st.subheader("Analyst Recommendations")
            analyst_recommendations = finance_agent.run(f"Summarize analyst recommendations for {symbol}")
            recommendations_data = analyst_recommendations.messages[-1].content  # Extract raw data
            recommendations = parse_raw_data(recommendations_data)

            # Fetch company news
            st.subheader("Latest News")
            company_news = finance_agent.run(f"Fetch latest news for {symbol}")
            news_data = company_news.messages[-1].content  # Extract raw data
            news = parse_raw_data(news_data)
            
            # Fetch balance sheet
            st.subheader("Balance Sheet")
            balance_sheet = fetch_balance_sheet(symbol)
            if not balance_sheet.empty:
                st.write(balance_sheet)
            else:
                st.warning("No balance sheet data available.")
            # Fetch financial metrics
            st.subheader("Financial Metrics")
            financial_metrics = fetch_financial_metrics(symbol)
            st.write("### Key Financial Metrics")
            st.table(pd.DataFrame.from_dict(financial_metrics, orient="index", columns=["Value"]))

            # Fetch past events
            st.subheader("Past Events")
            past_events = fetch_past_events(symbol)
            st.write("### Earnings Dates")
            if not past_events["Earnings Dates"].empty:
                st.write(past_events["Earnings Dates"])
            else:
                st.warning("No earnings dates available.")

            st.write("### Stock Splits")
            if not past_events["Stock Splits"].empty:
                st.write(past_events["Stock Splits"])
            else:
                st.warning("No stock splits available.")

            # Generate future outlook
            print(news_data)
            st.subheader("Future Outlook")
            future_outlook = None  # Initialize with a default value
            if recommendations_data and news_data and not balance_sheet.empty and financial_metrics:
                future_outlook = generate_future_outlook(
                    symbol=symbol,
                    news=news_data,
                    balance_sheet=balance_sheet,
                    financial_metrics=financial_metrics,
                    analyst_recommendations=recommendations_data
                )
                st.write(future_outlook)
            else:
                st.error("Insufficient data to generate future outlook.")
                future_outlook = "No future outlook available due to insufficient data."  # Assign a placeholder message

            # Store all data in ChromaDB
            stock_data = {
                "analyst_recommendations": recommendations_data,
                "news": news_data,
                "balance_sheet": balance_sheet,
                "financial_metrics": financial_metrics,
                "past_events": {
                    "earnings_dates": past_events["Earnings Dates"],
                    "stock_splits": past_events["Stock Splits"]
                },
                "future_outlook": future_outlook  # Now this variable is always assigned
            }

            try:
                store_data_in_chromadb(symbol, stock_data)
                st.success("Successfully stored data in ChromaDB")
            except Exception as e:
                st.error(f"Failed to store data: {str(e)}")
    with tab2:
        create_chat_interface()

# Run the app
if __name__ == "__main__":
    main()