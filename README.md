# 📊 AI-Powered Stock Analysis and Chat Application

This is a Streamlit-based application for analyzing stock data and interacting with a smart assistant that utilizes multiple AI agents powered by Google's Gemini models. It fetches financial metrics, balance sheets, past events, company news, and analyst recommendations, and uses that data to generate future outlooks and answer user queries.

## 🚀 Features

- **Stock Symbol Input:** Analyze any publicly traded company (e.g., NVDA, AAPL).
- **Multi-Agent AI System:**
  - Web Search Agent (powered by DuckDuckGo)
  - Finance AI Agent (powered by YFinance tools)
  - Combined Multi-AI Agent for complex queries
- **Data Analysis:**
  - Analyst recommendations
  - Latest company news (with sentiment analysis using FinBERT)
  - Financial metrics (P/E ratio, market cap, etc.)
  - Balance sheet and past stock events
- **Future Outlook Generation:** Uses AI to generate a detailed outlook based on aggregated data.
- **ChromaDB Integration:** Stores and retrieves past stock data efficiently using embeddings.
- **Interactive Chatbot:** Ask follow-up questions about stored stock data.

## 🧠 Tech Stack

- `Streamlit` for UI
- `phi` AI framework for agent orchestration
- `Google Gemini API` for intelligent responses
- `YFinance` for financial data
- `ChromaDB` for vector-based document storage
- `Transformers` (`FinBERT`) for sentiment analysis
- `.env` for managing API keys and environment variables

## 📦 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
GOOGLE_API_KEY=your_google_generative_ai_key
streamlit run stock_analysis.py


