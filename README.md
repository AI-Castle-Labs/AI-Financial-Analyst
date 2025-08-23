# ai-finance-analyst

Agentic AI financial analyst system using LangGraph, LangChain, OpenAI, and modular agent schemas.

## Features
- Modular agent workflow: macro, sector, central bank, FX, portfolio manager, etc.
- PDF report generation
- Plug-and-play with external data sources (FRED, IMF, ECB, World Bank)
- Slack/X/Twitter posting (via tweepy)
- Deep search, ranking, and LLM-based similarity scoring
- OpenAI embeddings (v1 API)
- System planner agent for graph-based research

## Installation
```bash
pip install ai-finance-analyst
```

## Usage
```python
from ai_finance_analyst import DeepResearchAgent
agent = DeepResearchAgent()
agent.run("Conduct research on equity valuations across the US")
```

## API
- `DeepResearchAgent.run(prompt: str)` — Run the full agentic workflow and generate a PDF report.
- See `finance/ai_finance_analyst.py` for more details and agent customization.

## Environment
- Requires Python 3.9+
- Set your `OPENAI_API_KEY` and other secrets in a `.env` file.

## License
MIT
