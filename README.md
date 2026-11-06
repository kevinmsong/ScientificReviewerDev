# ScientificReviewerDev

Development / experimental fork of the Scientific Reviewer Streamlit app: multi-expert AI peer review of uploaded PDFs (papers, grants, posters) with dialogue, moderation, and PDF export.

## What’s different from ScientificReviewer

- In-app caption: **v2.4.0**
- Extra model option in the UI: **o3-mini** (alongside GPT-4o and Gemini 2.0 Flash)
- Global **Review Style** slider applied to all reviewers (instead of per-reviewer style)
- Token-aware content chunking (`tiktoken`) before sending long PDFs to models

## Features

- PDF text extraction (PyMuPDF)
- Configurable reviewers, iterations, optional moderator
- Rating scales: Paper Score (−2 to 2), Star Rating (1–5), NIH Scale (1–9)
- Expert dialogue across iterations and downloadable ReportLab PDF summary

## Requirements

See `requirements.txt` (streamlit, openai, langchain, langchain-community, PyMuPDF, Pillow, tiktoken, google-generativeai, reportlab).

## Setup

```bash
pip install -r requirements.txt
```

Configure Streamlit secrets:

```toml
openai_api_key = "YOUR_OPENAI_API_KEY"
gemini_api_key = "YOUR_GEMINI_API_KEY"
```

## Usage

```bash
streamlit run ScientificReviewerDev.py
```
