from setuptools import setup, find_packages

setup(
    name="codechat-ai",
    version="1.0.0",
    packages=find_packages(),
    py_modules=['cli'],
    install_requires=[
        "fastapi==0.109.0",
        "uvicorn[standard]==0.27.0",
        "faiss-cpu",
        "sentence-transformers>=2.3.0",
        "google-generativeai==0.8.6",
        "python-dotenv==1.0.0",
        "langchain-text-splitters>=0.2.0",
        "pydantic>=2.5.3",
        "rich"
    ],
    entry_points={
        'console_scripts': [
            'codechat=cli:main',
        ],
    },
)
