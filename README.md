# Medical and Cybersecurity RAG Systems

**Note: This project is designed to raise awareness about security concerns in RAG systems. Some components demonstrate intentionally flawed implementations.**

This repository contains multiple Retrieval-Augmented Generation (RAG) systems for medical and cybersecurity applications. It showcases both good practices and potential security risks in RAG implementations.

## Overview

These RAG systems aim to demonstrate the potential and pitfalls of using AI for sensitive domains like healthcare and cybersecurity. The project includes secure implementations as well as intentionally flawed systems to highlight common security and privacy risks.

## Features

- Good and bad implementations of medical and cybersecurity RAG systems
- Integration with OpenAI's GPT models
- Document processing pipeline for various file types
- FastAPI-based API endpoints
- Demonstration of privacy and security risks in flawed implementations

## Technical Stack

- Backend: Python
- API Framework: FastAPI
- NLP Models: OpenAI GPT-3.5 and GPT-4
- Document Processing: Haystack
- Embeddings: SentenceTransformers

## File Structure

- `good_medical_rag.py`: Secure implementation of a medical RAG system
- `good_cyber_rag.py`: Secure implementation of a cybersecurity RAG system
- `bad_medical_rag.py`: Flawed medical RAG system demonstrating privacy risks
- `bad_cyber_rag.py`: Flawed cybersecurity RAG system using unreliable data
- `requirements.txt`: List of project dependencies

## Setup

1. Clone the repository
2. Create a virtual environment and activate it
3. Install dependencies: `pip install -r requirements.txt`
4. Set up a `.env` file with your OpenAI API key
5. Run the desired system using Uvicorn, e.g., `uvicorn good_medical_rag:app --reload`

## Usage

Access the API at `http://localhost:8000` and use the `/question/{question}` endpoint to interact with the RAG system.

## Demo

To see a demo, run one of the systems and send a request to the `/question/{question}` endpoint.

## Acknowledgements

- OpenAI for providing the GPT models
- Haystack for document processing capabilities
- FastAPI for the API framework

## Security Awareness

This project aims to raise awareness about several security concerns in RAG systems:

1. Data Privacy: The bad medical RAG system demonstrates how improper data handling can lead to privacy breaches.
2. Information Reliability: The bad cybersecurity RAG system shows the dangers of using unreliable or misleading information sources.
3. Model Behavior: Different responses from GPT-3.5 and GPT-4 highlight the importance of model selection and prompt engineering.
4. Access Control: The systems demonstrate the need for proper access controls to sensitive information.

By exploring these systems, developers can better understand the potential risks and best practices in implementing RAG systems for sensitive domains.
