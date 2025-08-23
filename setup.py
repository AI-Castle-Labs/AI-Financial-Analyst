from setuptools import setup, find_packages

setup(
    name="ai-finance-analyst",
    version="0.1.0",
    description="Agentic AI financial analyst system using LangGraph, LangChain, OpenAI, and modular agent schemas.",
    author="AI Castle Labs",
    author_email="your@email.com",
    packages=find_packages(),
    install_requires=[
        "openai>=1.3.0",
        "langchain>=0.2.10",
        "langgraph>=0.2.14",
        "langchain-openai>=0.2.0",
        "python-dotenv>=1.0.1",
        "numpy>=1.26.0",
        "pydantic>=2.7.0",
        "tweepy>=4.14.0",
        "fpdf2>=2.7.9",
        "tavily-python>=0.3.3"
    ],
    python_requires='>=3.9',
    include_package_data=True,
    url="https://github.com/yourusername/ai-finance-analyst",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
