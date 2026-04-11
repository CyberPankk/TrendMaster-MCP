from mcp.server.fastmcp import FastMCP
import httpx
import asyncio
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

mcp = FastMCP("TrendMaster-Research-MCP")

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import PyPDF2
import docx
import io
import os

@mcp.tool()
async def extract_youtube_transcript(video_id: str) -> str:
    """Extract English transcript from a YouTube video for quantitative research."""
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=['en'])
        texts = [item['text'].strip() for item in transcript_list]
        full_text = " ".join(texts)
        return full_text[:10000] # Return up to 10000 chars to avoid token limits
    except TranscriptsDisabled:
        return f"Error: Transcripts are disabled for video {video_id}."
    except NoTranscriptFound:
        return f"Error: No English transcript found for video {video_id}."
    except Exception as e:
        return f"Error extracting transcript: {e}"

@mcp.tool()
async def parse_local_document(file_path: str) -> str:
    """Parse text from local PDF, Word (.docx), or Markdown/Text files."""
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
        
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in ['.md', '.txt']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()[:10000]
        elif ext == '.pdf':
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text[:10000]
        elif ext == '.docx':
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
            return text[:10000]
        else:
            return f"Error: Unsupported file format {ext}"
    except Exception as e:
        return f"Error reading document: {e}"


@mcp.tool()
async def search_academic_papers(query: str, max_results: int = 3) -> str:
    """Search arXiv for quantitative trading papers."""
    url = f'http://export.arxiv.org/api/query?search_query=all:"{query}"&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=desc'
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            return f"Error: arXiv API returned status code {response.status_code}"
            
        root = ET.fromstring(response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        results = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            results.append(f"Title: {title}\nAbstract: {summary}\n---")
            
        return "\n".join(results) if results else "No papers found."

@mcp.tool()
async def extract_social_alpha(authors: list[str], keywords: list[str]) -> str:
    """Mock implementation of Twitter alpha extraction."""
    # In a real implementation, this would use the Twitter API
    mock_tweets = [
        f"[{authors[0]}] Noticeable divergence between funding rates and price action. Usually precedes a volatility spike.",
        f"[{authors[0]}] Open interest just hit an all-time high while spot volume is declining. Be careful."
    ]
    
    # Filter by keywords
    filtered = []
    for tweet in mock_tweets:
        if any(kw.lower() in tweet.lower() for kw in keywords):
            filtered.append(tweet)
            
    return "\n".join(filtered) if filtered else "No relevant alpha signals found."

if __name__ == "__main__":
    mcp.run()