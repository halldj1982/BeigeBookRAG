# BeigeBook RAG Assistant — Streamlit + OpenSearch (Terraform)

This updated repository adds Terraform to provision an AWS OpenSearch (k-NN) domain, S3 bucket for PDFs,
and IAM resources. It also replaces the FAISS demo with a real OpenSearch vector store and provides
two Streamlit pages:
- `pages/1_chatbot.py` — the RAG chatbot UI
- `pages/2_ingest.py` — upload PDFs and ingest into OpenSearch

**Important**: You must configure AWS credentials and Streamlit secrets before deploying.

## High-level steps
1. Use Terraform (in `terraform/`) to provision OpenSearch domain and S3 bucket.
2. Deploy Streamlit app to Streamlit Community Cloud, set secrets for AWS and OpenSearch.
3. Use the Ingest page to upload PDFs; they will be parsed, embeddings generated via Amazon Bedrock, and indexed into OpenSearch.
4. Use the Chatbot page to ask questions; the app performs vector search against OpenSearch and calls Claude Sonnet 3.7 via Bedrock for answer generation.

## Security notes
- Do NOT commit AWS credentials. Use Streamlit secrets or environment variables.
- Terraform will create IAM principals; review them and restrict as appropriate for production.
- The OpenSearch access policy defaults to a customizable CIDR (variable `allowed_cidr`). Set it to a safe value.

