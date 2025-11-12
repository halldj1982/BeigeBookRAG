# BeigeBook RAG Assistant — Streamlit + OpenSearch (Terraform)

This updated repository adds Terraform to provision an AWS OpenSearch (k-NN) domain, S3 bucket for PDFs,
and IAM resources. It also replaces the FAISS demo with a real OpenSearch vector store and provides
two Streamlit pages:
- `pages/1_chatbot.py` — the RAG chatbot UI
- `pages/2_ingest.py` — upload PDFs and ingest into OpenSearch

**Important**: You must configure AWS credentials and Streamlit secrets before deploying.

## High-level steps
1. Use Terraform (in `terraform/`) to provision OpenSearch domain and S3 bucket.
2. Deploy Streamlit app to AWS App Runner (see deployment section below) or Streamlit Community Cloud.
3. Use the Ingest page to upload PDFs; they will be parsed, embeddings generated via Amazon Bedrock, and indexed into OpenSearch.
4. Use the Chatbot page to ask questions; the app performs vector search against OpenSearch and calls Claude Sonnet 3.7 via Bedrock for answer generation.

## AWS App Runner Deployment (Git-based)

### Prerequisites
1. AWS CLI configured with appropriate permissions
2. GitHub repository with your code
3. Infrastructure deployed via Terraform (OpenSearch + S3)
4. GitHub connection established in App Runner (first-time setup)

### Step-by-Step Deployment

#### 1. Create App Runner Configuration
Create `apprunner.yaml` in the project root:
```yaml
version: 1.0
runtime: python3
build:
  commands:
    build:
      - echo "Build started on `date`"
      - pip install -r requirements.txt
run:
  runtime-version: 3.11
  command: streamlit run app.py --server.port=8501 --server.address=0.0.0.0
  network:
    port: 8501
    env: PORT
  env:
    - name: AWS_DEFAULT_REGION
      value: "us-west-2"
```

#### 2. Push Code to GitHub
```bash
# Initialize git repository (if not already done)
git init
git add .
git commit -m "Initial commit"

# Add GitHub remote and push
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
git branch -M main
git push -u origin main
```

#### 3. Create App Runner Service via AWS Console
1. Go to AWS App Runner console
2. Click "Create service"
3. **Source**: Choose "Source code repository"
4. **Connect to GitHub**: Authorize AWS to access your GitHub account
5. **Repository**: Select your repository
6. **Branch**: Choose `main` (or your default branch)
7. **Configuration**: Choose "Use a configuration file" (apprunner.yaml)
8. **Service name**: `beigebook-rag-app`
9. Click "Next" through the remaining steps and "Create & deploy"

#### 4. Configure Environment Variables (After Service Creation)
1. Go to your App Runner service in the AWS console
2. Click on the "Configuration" tab
3. In the "Environment variables" section, click "Edit"
4. Add the following environment variables:
   - `OPENSEARCH_ENDPOINT`: `https://your-opensearch-endpoint`
   - `OPENSEARCH_INDEX`: `beigebook-docs`
   - `CLAUDE_MODEL`: `amazon.nova-premier-v1:0`
   - `BEDROCK_EMBEDDING_MODEL`: `amazon.titan-embed-text-v2:0`
   - `EMBED_DIM`: `1024`
   - `S3_BUCKET`: `your-s3-bucket-name`
5. Click "Save changes" - this will trigger a new deployment

#### 5. Alternative: Create Service via CLI
```bash
# Create service configuration JSON
cat > apprunner-service.json << EOF
{
  "ServiceName": "beigebook-rag-app",
  "SourceConfiguration": {
    "CodeRepository": {
      "RepositoryUrl": "https://github.com/YOUR-USERNAME/YOUR-REPO-NAME",
      "SourceCodeVersion": {
        "Type": "BRANCH",
        "Value": "main"
      },
      "CodeConfiguration": {
        "ConfigurationSource": "CONFIGURATION_FILE"
      }
    },
    "AutoDeploymentsEnabled": true
  },
  "InstanceConfiguration": {
    "Cpu": "1 vCPU",
    "Memory": "2 GB"
  }
}
EOF

# Create the App Runner service
aws apprunner create-service --cli-input-json file://apprunner-service.json --region us-west-2
```

#### 6. Configure IAM Role
Create an IAM role for App Runner with permissions for:
- Amazon Bedrock (InvokeModel)
- OpenSearch (es:*)
- S3 (GetObject, PutObject)

#### 7. Update OpenSearch Access Policy
Add App Runner's IP ranges to your OpenSearch `allowed_cidr` or use IAM-based authentication.

#### 8. Monitor Deployment
```bash
# Check service status
aws apprunner describe-service --service-arn <service-arn> --region us-west-2

# View logs
aws logs describe-log-groups --log-group-name-prefix "/aws/apprunner/beigebook-rag-app"
```

### Environment Variables
Replace the `.streamlit/secrets.toml` approach with environment variables in App Runner:
- `OPENSEARCH_ENDPOINT`
- `OPENSEARCH_INDEX` 
- `CLAUDE_MODEL`
- `BEDROCK_EMBEDDING_MODEL`
- `EMBED_DIM`
- `S3_BUCKET`

### Cost Optimization
- Use App Runner's auto-scaling to handle traffic spikes
- Consider pausing the service during low-usage periods
- Monitor Bedrock and OpenSearch costs

## Security notes
- Do NOT commit AWS credentials. Use Streamlit secrets or environment variables.
- Terraform will create IAM principals; review them and restrict as appropriate for production.
- The OpenSearch access policy defaults to a customizable CIDR (variable `allowed_cidr`). Set it to a safe value.

