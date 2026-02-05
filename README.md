# Customer-Feedback-Intelligence
AI-powered system for organizing unstructured customer feedback into actionable insights


**Customer Feedback Intelligence System**

AI-Powered Insight Extraction from Unstructured Customer Feedback

Customer Feedback Intelligence System is an end-to-end analytics application that transforms raw, messy customer feedback into organized themes, trends, and actionable product insights using AI and data analytics techniques.

This project simulates how modern product, analytics, and CX teams analyze large volumes of unstructured feedback to inform decision-making.


**Why I Built This**

Customer feedback often comes from many sources—app reviews, surveys, support tickets, emails, and social media—and is rarely clean or consistently structured.

As a Management Information Systems graduate, I wanted to build a system that:

-Accepts real-world messy data

-Does not require a rigid template

-Automatically organizes feedback into insights that product and business teams can act on

This project focuses on bridging the gap between unstructured text data and decision-ready insights.


**What This Product Does**

The Customer Feedback Intelligence System allows users to:

-Upload any CSV file containing customer feedback (structured or unstructured)

-Map columns dynamically (text is required; metadata is optional)

-Normalize messy data into a consistent internal schema

-Automatically detect:
  -recurring themes
  -sentiment patterns
  -emerging trends over time

-Generate executive-level summaries and recommended actions

-All analysis runs instantly in-memory, making it ideal for fast exploration and demos.


**Key Features**

-Accepts any CSV format

-Supports:
  -one-column raw text files
  -structured app review datasets
  -support ticket exports

-Column-mapping wizard allows users to specify:
  -feedback text (required)
  -date, rating, and source (optional)


**Automatic Data Normalization**

Uploaded data is converted into a standard internal schema:

id | date | source | rating | text


This enables consistent downstream analysis without forcing users to reformat their data.


**Theme Detection & Clustering**

Feedback text is converted into semantic embeddings using OpenAI
Embeddings are clustered to identify recurring topics and patterns
Each cluster represents a feedback theme (e.g., billing, login issues, notifications)


**Interactive Dashboards**

-Sentiment distribution
-Top themes by priority
-Theme trends over time
-Deep-dive views with example comments


**Architecture Overview**

CSV Upload (Any Format)--> 
Column Mapping (User-defined)--> 
Data Normalization--> 
In-Memory Processing (Pandas)--> 
Text Embeddings (OpenAI)--> 
Clustering & Theme Detection--> 
AI Theme Labeling--> 
Dashboards & Executive Insights


**Languages & Tools**

-Python

-Pandas

-NumPy

-AI & NLP

-OpenAI API (Embeddings + LLM-based labeling)

-Visualization & UI

-Streamlit

-Plotly

-VS Code

-GitHub

**Demo Video:** (used sample data already provided)

https://github.com/user-attachments/assets/aa8ed60e-e3e2-47a7-92cb-df7ae3486c5a

