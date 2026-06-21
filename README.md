# graphScholar: An Automated Pipeline for Conversational Exploration of Scientific Literature via Knowledge Graphs and Retrieval-Augmented Generation

## Executive Summary

GraphScholar is an automated pipeline that transforms a user-supplied research topic into an interactive, conversational knowledge base grounded in current scientific literature. Given a topic, the system retrieves relevant papers from arXiv, extracts structured entities and relationships (methods, findings, authors, datasets) using an LLM, and assembles these extractions into a knowledge graph. The graph is indexed using GraphRAG techniques — including community detection to generate cluster-level summaries — enabling retrieval of both fine-grained facts and higher-level thematic context.

Users interact with the system conversationally: asking questions about a topic returns synthesized, citation-grounded answers drawn from the underlying graph and source papers, rather than flat document retrieval. This demonstrates an end-to-end architecture spanning literature ingestion, automated information extraction, graph construction, and graph-based retrieval-augmented generation — offering a scalable approach to navigating and querying large bodies of research literature through natural conversation.

## Project Structure

```
graphscholar/
├── src/          # Pipeline source code (ingestion, extraction, graph, RAG, chat)
├── data/         # Raw and processed paper data, state tracking DB
├── docs/         # Design docs, schema definitions, write-up
└── notebooks/    # Exploration and prototyping
```

## Status

🚧 In development — summer project.
