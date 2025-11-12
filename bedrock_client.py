# bedrock_client.py
"""
Robust Bedrock client wrapper.

This wrapper attempts to call the Bedrock 'InvokeModel' operation in a way that is:
 - tolerant to boto3/botocore version differences,
 - robustly logs diagnostics if the operation is missing,
 - returns a consistent dict with 'output' or raises a helpful error.

Usage:
  client = BedrockClient(config)
  client.generate(prompt=..., model="anthropic.claude-sonnet-3.7")
  client.embed(text=..., model="amazon.titan-embed-text-v2")
"""

import boto3
import json
import sys
from typing import Dict, Any


class BedrockClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        region = self.config.get('aws_region') or None
        # Create client; allow region override from config, else boto3 default
        if region:
            self.client = boto3.client('bedrock-runtime', region_name=region)
        else:
            self.client = boto3.client('bedrock-runtime')
        # figure out which operation name is available
        self.operations = []
        try:
            self.operations = list(self.client.meta.service_model.operation_names)
        except Exception:
            # older boto3/botocore might not expose operation_names; fallback to dir()
            self.operations = [m for m in dir(self.client) if not m.startswith('_')]

        # canonical op name we want
        self._invoke_op = None
        for cand in ("InvokeModel", "invoke_model", "invokeModel"):
            if any(cand.lower() == op.lower() for op in self.operations):
                self._invoke_op = cand
                break

    def _invoke(self, modelId: str, body: Dict[str, Any], contentType: str = "application/json", accept: str = "application/json"):
        """
        Attempt to call an available 'invoke' operation (InvokeModel / invoke_model).
        Returns the raw HTTP body (string) on success.
        """
        if not self._invoke_op:
            # Provide diagnostics and a helpful error
            msg = (
                "Bedrock 'InvokeModel' operation not available in boto3 client.\n"
                "Diagnosed operations: {}\n\n"
                "Possible fixes:\n"
                " - Upgrade boto3 & botocore in your virtualenv: pip install --upgrade boto3 botocore\n"
                " - If your environment cannot call Bedrock directly, run a small proxy (Lambda/API Gateway) in AWS and call that instead.\n"
                " - Ensure your boto3 is recent enough to include 'bedrock' service support.\n"
            ).format(self.operations)
            raise RuntimeError(msg)

        # Call using the low-level client method name if present, else use meta.invoke
        op_name = self._invoke_op
        # normalize to lower-case method name if required (boto3 exposes snake_case methods)
        method = None
        # prefer attribute if exists
        if hasattr(self.client, op_name):
            method = getattr(self.client, op_name)
            try:
                # common shape: client.invoke_model(modelId=..., contentType='application/json', accept='application/json', body=json.dumps({...}))
                resp = method(modelId=modelId, contentType=contentType, accept=accept, body=json.dumps(body))
                return resp
            except TypeError:
                # method exists but signature differs; fall through to generic call
                pass
        # fallback: use client._make_api_call (botocore internal) with operation name mapping
        try:
            # botocore operation names are typically "InvokeModel"
            return self.client._make_api_call(self._invoke_op, {
                "modelId": modelId,
                "contentType": contentType,
                "accept": accept,
                "body": json.dumps(body)
            })
        except Exception as e:
            # raise with context
            raise RuntimeError(f"Failed to call Bedrock invoke operation '{self._invoke_op}': {e}")

    def generate(self, prompt: str, model: str = None, max_tokens: int = 1024) -> Dict[str, Any]:
        """
        High-level text generation helper. Returns {'output': <text>} or raises.
        """
        model = model or self.config.get('claude_model')
        if not model:
            raise RuntimeError("Claude model not configured (pass model or set 'claude_model' in config).")

        # Format request body based on model type
        if "claude" in model.lower() or "anthropic" in model.lower():
            # Check if it's Claude 3.7 which requires Messages API format
            if "claude-3-7" in model.lower():
                body = {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "anthropic_version": "bedrock-2023-05-31"
                }
            else:
                # Legacy Claude format
                body = {
                    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": max_tokens,
                    "temperature": 0.1,
                    "top_p": 0.9
                }
        elif "nova" in model.lower():
            # Amazon Nova format
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
        else:
            body = {
                "input": prompt,
                "max_tokens": max_tokens
            }
        resp = self._invoke(modelId=model, body=body)
        # resp shape depends on botocore/bedrock; try to parse common variations
        try:
            # resp may be a dict already (if _make_api_call used)
            if isinstance(resp, dict):
                # look for keys containing text content
                if "body" in resp and hasattr(resp["body"], "read"):
                    raw = resp["body"].read().decode("utf-8")
                else:
                    raw = json.dumps(resp)
            else:
                # resp might be a requests-like response; attempt to read
                raw = None
                if hasattr(resp, "get") and "body" in resp:
                    raw = resp["body"]
                if raw is None:
                    raw = str(resp)
            # try to parse JSON
            try:
                parsed = json.loads(raw)
                # standardize returning the parsed content under 'output' where possible
                # Different models return text in different keys
                if isinstance(parsed, dict):
                    # Amazon Nova format
                    if "output" in parsed and "message" in parsed["output"]:
                        message = parsed["output"]["message"]
                        if "content" in message and isinstance(message["content"], list) and len(message["content"]) > 0:
                            if "text" in message["content"][0]:
                                return {"output": message["content"][0]["text"]}
                    # Claude 3.7 Messages API format
                    if "content" in parsed and isinstance(parsed["content"], list) and len(parsed["content"]) > 0:
                        if "text" in parsed["content"][0]:
                            return {"output": parsed["content"][0]["text"]}
                    # Legacy Claude models return text under 'completion'
                    if "completion" in parsed:
                        return {"output": parsed["completion"]}
                    # Other models might use 'content' or 'outputText'
                    if "content" in parsed:
                        return {"output": parsed["content"]}
                    if "outputText" in parsed:
                        return {"output": parsed["outputText"]}
                return {"output": parsed}
            except Exception:
                # not JSON – return as string
                return {"output": raw}
        except Exception as e:
            raise RuntimeError(f"Failed to parse Bedrock response: {e}")

    def embed(self, text: str, model: str = None) -> Dict[str, Any]:
        """
        Embedding helper. Returns {'embedding': [...]} on success.
        """
        model = model or self.config.get('bedrock_embedding_model')
        if not model:
            raise RuntimeError("Embedding model not configured (set 'bedrock_embedding_model').")
        
        # Format request body based on model type
        if "titan" in model.lower():
            body = {"inputText": text}
        else:
            body = {"input": text}
            
        resp = self._invoke(modelId=model, body=body)
        # parse response similar to generate()
        try:
            if isinstance(resp, dict):
                if "body" in resp and hasattr(resp["body"], "read"):
                    raw = resp["body"].read().decode("utf-8")
                else:
                    raw = json.dumps(resp)
            else:
                raw = str(resp)
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                raise RuntimeError("Bedrock embed response not JSON or unexpected format")
            
            # Titan models return embedding directly under 'embedding' key
            if isinstance(parsed, dict) and "embedding" in parsed:
                return {"embedding": parsed["embedding"]}
            
            raise RuntimeError("Could not find embedding vector in Bedrock response")
        except Exception as e:
            raise RuntimeError(f"Failed to parse embed response: {e}")
