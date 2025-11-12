import streamlit as st
import os
from pathlib import Path
import toml

def load_config():
    cfg = {}
    # 1) st.secrets: support both flat keys and [default] table
    try:
        # If top-level keys exist, pick them
        for key in ("opensearch_endpoint","aws_region","claude_model","bedrock_embedding_model","opensearch_index","embed_dim","chunk_size","max_rounds","s3_bucket"):
            val = st.secrets.get(key)
            if val is not None:
                cfg[key] = val
        # If there's a 'default' table, merge it (but don't overwrite keys already set)
        default_table = st.secrets.get("default")
        if isinstance(default_table, dict):
            for k,v in default_table.items():
                if k not in cfg:
                    cfg[k] = v
    except Exception:
        pass

    # 2) local secrets.toml in project root (fallback)
    p = Path("secrets.toml")
    if p.exists():
        try:
            parsed = toml.loads(p.read_text())
            # toml may return a dict with 'default' table — handle that too
            if "default" in parsed and isinstance(parsed["default"], dict):
                parsed = {**parsed["default"], **{k:v for k,v in parsed.items() if k!="default"}}
            for k,v in parsed.items():
                if k not in cfg:
                    cfg[k] = v
        except Exception:
            pass

    # 3) environment variables fallback
    env_map = {
        "opensearch_endpoint": ["OPENSEARCH_ENDPOINT","opensearch_endpoint"],
        "opensearch_index": ["OPENSEARCH_INDEX","opensearch_index"],
        "aws_region": ["AWS_REGION","aws_region"],
        "claude_model": ["CLAUDE_MODEL","claude_model"],
        "bedrock_embedding_model": ["BEDROCK_EMBEDDING_MODEL","bedrock_embedding_model"],
        "embed_dim": ["EMBED_DIM","embed_dim"],
        "chunk_size": ["CHUNK_SIZE","chunk_size"],
        "max_rounds": ["MAX_ROUNDS","max_rounds"],
        "s3_bucket": ["S3_BUCKET","s3_bucket"]
    }
    for cfg_key, env_keys in env_map.items():
        if cfg_key not in cfg:
            for envk in env_keys:
                if os.environ.get(envk) is not None:
                    cfg[cfg_key] = os.environ.get(envk)
                    break

    # Type conversions / defaults
    if "embed_dim" in cfg:
        try:
            cfg["embed_dim"] = int(cfg["embed_dim"])
        except Exception:
            pass
    cfg["opensearch_index"] = cfg.get("opensearch_index", "beigebook-docs")
    cfg["max_rounds"] = int(cfg.get("max_rounds", 3))
    cfg["chunk_size"] = int(cfg.get("chunk_size", 800))

    return cfg
