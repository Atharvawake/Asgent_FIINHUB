import os
import requests
from langchain_core.tools import tool

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

def _get_api_key():
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        raise ValueError("FINNHUB_API_KEY is missing from environment variables.")
    return key

@tool
def get_stock_quote(symbol: str) -> str:
    """
    Fetch real-time stock price, day high/low, open price, and previous close 
    for a given stock symbol/ticker (e.g., AAPL, NVDA, TSLA, GOOGL, MSFT).
    """
    try:
        api_key = _get_api_key()
        ticker = symbol.upper().strip()
        url = f"{FINNHUB_BASE_URL}/quote?symbol={ticker}&token={api_key}"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        current = data.get("c")
        if not current:
            return f"Could not find stock quote for ticker '{ticker}'."
            
        high = data.get("h")
        low = data.get("l")
        open_price = data.get("o")
        prev_close = data.get("pc")
        change = data.get("d")
        percent_change = data.get("dp", 0)

        return (
            f"📈 Stock Data for {ticker}:\n"
            f"- Current Price: ${current}\n"
            f"- Change: {change} ({percent_change:.2f}%)\n"
            f"- Day High: ${high}\n"
            f"- Day Low: ${low}\n"
            f"- Open Price: ${open_price}\n"
            f"- Previous Close: ${prev_close}"
        )
    except Exception as e:
        return f"Error fetching stock data for {symbol}: {str(e)}"


@tool
def get_company_profile(symbol: str) -> str:
    """
    Fetch company background information, market capitalization, industry, 
    web URL, and exchange info for a stock ticker.
    """
    try:
        api_key = _get_api_key()
        ticker = symbol.upper().strip()
        url = f"{FINNHUB_BASE_URL}/stock/profile2?symbol={ticker}&token={api_key}"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if not data or "name" not in data:
            return f"No profile found for ticker '{ticker}'."
            
        return (
            f"🏢 Company Profile: {data.get('name')} ({ticker})\n"
            f"- Industry: {data.get('finnhubIndustry')}\n"
            f"- Market Cap: ${data.get('marketCapitalization', 0):,.2f} M\n"
            f"- Country: {data.get('country')}\n"
            f"- Currency: {data.get('currency')}\n"
            f"- Exchange: {data.get('exchange')}\n"
            f"- Website: {data.get('weburl')}"
        )
    except Exception as e:
        return f"Error fetching company profile for {symbol}: {str(e)}"


@tool
def get_market_news(category: str = "general") -> str:
    """
    Fetch top headlines/news for global financial markets. 
    Categories can be: 'general', 'forex', 'crypto', or 'merger'.
    """
    try:
        api_key = _get_api_key()
        url = f"{FINNHUB_BASE_URL}/news?category={category}&token={api_key}"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if not data:
            return "No recent market news found."
            
        news_items = []
        for article in data[:5]:
            headline = article.get("headline")
            source = article.get("source")
            summary = article.get("summary")
            url_link = article.get("url")
            news_items.append(f"• {headline} ({source})\n  Summary: {summary}\n  Link: {url_link}")
            
        return "\n\n".join(news_items)
    except Exception as e:
        return f"Error fetching market news: {str(e)}"

# Consolidated tool list for export
finance_tools = [get_stock_quote, get_company_profile, get_market_news]