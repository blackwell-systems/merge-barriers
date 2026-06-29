#!/usr/bin/env python3
"""
Evaluate structok model on structured data comprehension.

Compares the model's ability to predict next tokens in JSON vs GCF
encodings of identical data. Lower perplexity = better comprehension.

Supports three modes:
  1. Single model eval (original)
  2. Controlled comparison: two models, same test data, different tokenizers
  3. Pythia baseline comparison

Usage:
  # Single model eval
  python eval_model.py --checkpoint checkpoints/step-20000/checkpoint.pt \
      --tokenizer structok-64k.json

  # Controlled comparison (run-002)
  python eval_model.py --checkpoint checkpoints/structok/checkpoint.pt \
      --tokenizer structok-64k.json \
      --compare-checkpoint checkpoints/standard/checkpoint.pt \
      --compare-tokenizer standard-64k.json \
      --test-data /workspace/test_data/

  # Quick test (small payloads only)
  python eval_model.py --checkpoint checkpoints/step-20000/checkpoint.pt \
      --tokenizer structok-64k.json --quick
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch


# Same model configs as train_model.py
MODEL_CONFIGS = {
    "test": {
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "max_position_embeddings": 2048,
    },
    "125m": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 2048,
    },
    "410m": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "max_position_embeddings": 2048,
    },
}


def generate_test_data(n_records=20, seed=42):
    """Generate identical test data in both JSON and GCF format."""
    random.seed(seed)
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park",
             "Eva Johansson", "Frank Mueller", "Grace Kim", "Henry Liu"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    tiers = ["free", "basic", "standard", "premium", "enterprise"]

    records = []
    for i in range(n_records):
        records.append({
            "orderId": f"ORD-{random.randint(10000, 99999)}",
            "customer": random.choice(names),
            "tier": random.choice(tiers),
            "status": random.choice(statuses),
            "total": round(random.uniform(10, 2000), 2),
        })

    # JSON encoding
    json_str = json.dumps(records, indent=2)

    # GCF encoding
    fields = "orderId,customer,tier,status,total"
    gcf_lines = [f"## orders [{n_records}]{{{fields}}}"]
    for r in records:
        gcf_lines.append(f"{r['orderId']}|{r['customer']}|{r['tier']}|{r['status']}|{r['total']}")
    gcf_str = "\n".join(gcf_lines)

    return records, json_str, gcf_str


def compute_perplexity(model, tokenizer, text, device="cpu", max_length=2048):
    """Compute perplexity of text under the model."""
    ids = tokenizer.encode(text).ids
    if len(ids) > max_length:
        ids = ids[:max_length]
    if len(ids) < 2:
        return float("inf")

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss.item()

    if math.isnan(loss) or math.isinf(loss):
        return float("inf")
    return math.exp(min(loss, 20))


def compute_next_token_accuracy(model, tokenizer, text, device="cpu", max_length=2048):
    """Measure how often the model correctly predicts the next token."""
    ids = tokenizer.encode(text).ids
    if len(ids) > max_length:
        ids = ids[:max_length]
    if len(ids) < 2:
        return 0.0, 0

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Compare predicted token (argmax) with actual next token
    predictions = logits[0, :-1].argmax(dim=-1)  # (seq_len - 1,)
    targets = input_ids[0, 1:]  # (seq_len - 1,)

    correct = (predictions == targets).sum().item()
    total = len(targets)

    return correct / total, total


def load_structok_model(checkpoint_path, size="410m", tokenizer_path=None):
    """Load a structok model from a train_model.py checkpoint."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()

    cfg = MODEL_CONFIGS[size].copy()
    cfg["vocab_size"] = vocab_size
    config = GPTNeoXConfig(**cfg)
    model = GPTNeoXForCausalLM(config)

    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    step = cp.get("step", 0)
    train_loss = cp.get("loss", 0)
    print(f"Loaded structok model from step {step} (training loss: {train_loss:.4f})")

    model.eval()
    return model, tok, step


def load_pythia_model():
    """Load Pythia-410M for comparison."""
    from transformers import GPTNeoXForCausalLM, AutoTokenizer

    print("Loading Pythia-410M...")
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-410m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    model.eval()
    print("Loaded Pythia-410M")
    return model, tokenizer


def load_test_files(test_dir):
    """Load held-out test files from disk. Returns list of (n_records, json_str, gcf_str)."""
    test_dir = Path(test_dir)
    results = []
    for n in [5, 10, 20, 50, 100]:
        json_path = test_dir / f"test-{n}-records.json"
        gcf_path = test_dir / f"test-{n}-records.gcf"
        if not json_path.exists() or not gcf_path.exists():
            print(f"  Warning: test files for {n} records not found, skipping")
            continue
        json_str = json_path.read_text()
        gcf_str = gcf_path.read_text()
        results.append((n, json_str, gcf_str))
    return results


def eval_model_on_data(model, tok, test_data, device, label):
    """Evaluate a model on test data, return results list."""
    print()
    print(f"{'Records':<10} {'JSON PPL':>10} {'GCF PPL':>10} {'JSON Acc':>10} {'GCF Acc':>10} {'Winner':>10}")
    print("-" * 60)

    results = []
    for n, json_str, gcf_str in test_data:
        json_ppl = compute_perplexity(model, tok, json_str, device)
        gcf_ppl = compute_perplexity(model, tok, gcf_str, device)
        json_acc, json_total = compute_next_token_accuracy(model, tok, json_str, device)
        gcf_acc, gcf_total = compute_next_token_accuracy(model, tok, gcf_str, device)

        winner = "GCF" if gcf_ppl < json_ppl else "JSON"
        print(f"{n:<10} {json_ppl:>10.1f} {gcf_ppl:>10.1f} {json_acc:>9.1%} {gcf_acc:>9.1%} {winner:>10}")

        results.append({
            "records": n,
            "json_ppl": json_ppl,
            "gcf_ppl": gcf_ppl,
            "json_acc": json_acc,
            "gcf_acc": gcf_acc,
            "json_tokens": len(tok.encode(json_str).ids),
            "gcf_tokens": len(tok.encode(gcf_str).ids),
        })

    return results


def get_extended_test_data():
    """Return test data for extended evals across multiple formats and data types."""

    tests = {}

    # --- 1. Graph data (GCF symbols + edges vs JSON equivalent) ---
    tests["graph_10sym"] = {
        "label": "Graph (10 symbols, 8 edges)",
        "gcf": """## symbols [10]{id,kind,qname,score,provenance}
@0|function|auth.validate|0.95|definition
@1|class|api.Handler|0.88|definition
@2|method|db.connect|0.72|ast_inferred
@3|interface|service.Config|0.91|definition
@4|function|utils.parse|0.65|reference
@5|variable|cache.ttl|0.45|structural
@6|function|auth.refresh|0.82|definition
@7|method|api.respond|0.78|ast_inferred
@8|type|db.Schema|0.93|definition
@9|function|utils.encode|0.71|reference

## edges [8]{target,source,type}
@1<@0|calls
@2<@1|imports
@3<@1|implements
@4<@0|calls
@6<@0|calls
@7<@1|contains
@8<@2|references
@9<@4|calls""",
        "json": json.dumps({
            "symbols": [
                {"id": 0, "kind": "function", "qname": "auth.validate", "score": 0.95, "provenance": "definition"},
                {"id": 1, "kind": "class", "qname": "api.Handler", "score": 0.88, "provenance": "definition"},
                {"id": 2, "kind": "method", "qname": "db.connect", "score": 0.72, "provenance": "ast_inferred"},
                {"id": 3, "kind": "interface", "qname": "service.Config", "score": 0.91, "provenance": "definition"},
                {"id": 4, "kind": "function", "qname": "utils.parse", "score": 0.65, "provenance": "reference"},
                {"id": 5, "kind": "variable", "qname": "cache.ttl", "score": 0.45, "provenance": "structural"},
                {"id": 6, "kind": "function", "qname": "auth.refresh", "score": 0.82, "provenance": "definition"},
                {"id": 7, "kind": "method", "qname": "api.respond", "score": 0.78, "provenance": "ast_inferred"},
                {"id": 8, "kind": "type", "qname": "db.Schema", "score": 0.93, "provenance": "definition"},
                {"id": 9, "kind": "function", "qname": "utils.encode", "score": 0.71, "provenance": "reference"},
            ],
            "edges": [
                {"target": 1, "source": 0, "type": "calls"},
                {"target": 2, "source": 1, "type": "imports"},
                {"target": 3, "source": 1, "type": "implements"},
                {"target": 4, "source": 0, "type": "calls"},
                {"target": 6, "source": 0, "type": "calls"},
                {"target": 7, "source": 1, "type": "contains"},
                {"target": 8, "source": 2, "type": "references"},
                {"target": 9, "source": 4, "type": "calls"},
            ]
        }, indent=2),
    }

    tests["graph_20sym"] = {
        "label": "Graph (20 symbols, 15 edges)",
        "gcf": """## symbols [20]{id,kind,qname,score,provenance}
@0|function|auth.validate|0.95|definition
@1|class|api.Handler|0.88|definition
@2|method|db.connect|0.72|ast_inferred
@3|interface|service.Config|0.91|definition
@4|function|utils.parse|0.65|reference
@5|variable|cache.ttl|0.45|structural
@6|function|auth.refresh|0.82|definition
@7|method|api.respond|0.78|ast_inferred
@8|type|db.Schema|0.93|definition
@9|function|utils.encode|0.71|reference
@10|function|middleware.cors|0.84|definition
@11|class|router.Engine|0.92|definition
@12|method|logger.write|0.67|ast_inferred
@13|interface|store.Backend|0.89|definition
@14|function|crypto.hash|0.76|reference
@15|variable|config.port|0.41|structural
@16|function|queue.enqueue|0.81|definition
@17|method|cache.invalidate|0.73|ast_inferred
@18|type|event.Payload|0.87|definition
@19|function|metrics.record|0.69|reference

## edges [15]{target,source,type}
@1<@0|calls
@2<@1|imports
@3<@1|implements
@4<@0|calls
@6<@0|calls
@7<@1|contains
@8<@2|references
@9<@4|calls
@11<@10|calls
@12<@11|contains
@13<@11|implements
@14<@6|calls
@16<@7|calls
@17<@16|calls
@18<@12|references""",
        "json": json.dumps({
            "symbols": [
                {"id": i, "kind": k, "qname": q, "score": s, "provenance": p}
                for i, k, q, s, p in [
                    (0, "function", "auth.validate", 0.95, "definition"),
                    (1, "class", "api.Handler", 0.88, "definition"),
                    (2, "method", "db.connect", 0.72, "ast_inferred"),
                    (3, "interface", "service.Config", 0.91, "definition"),
                    (4, "function", "utils.parse", 0.65, "reference"),
                    (5, "variable", "cache.ttl", 0.45, "structural"),
                    (6, "function", "auth.refresh", 0.82, "definition"),
                    (7, "method", "api.respond", 0.78, "ast_inferred"),
                    (8, "type", "db.Schema", 0.93, "definition"),
                    (9, "function", "utils.encode", 0.71, "reference"),
                    (10, "function", "middleware.cors", 0.84, "definition"),
                    (11, "class", "router.Engine", 0.92, "definition"),
                    (12, "method", "logger.write", 0.67, "ast_inferred"),
                    (13, "interface", "store.Backend", 0.89, "definition"),
                    (14, "function", "crypto.hash", 0.76, "reference"),
                    (15, "variable", "config.port", 0.41, "structural"),
                    (16, "function", "queue.enqueue", 0.81, "definition"),
                    (17, "method", "cache.invalidate", 0.73, "ast_inferred"),
                    (18, "type", "event.Payload", 0.87, "definition"),
                    (19, "function", "metrics.record", 0.69, "reference"),
                ]
            ],
            "edges": [
                {"target": t, "source": s, "type": tp}
                for t, s, tp in [
                    (1, 0, "calls"), (2, 1, "imports"), (3, 1, "implements"),
                    (4, 0, "calls"), (6, 0, "calls"), (7, 1, "contains"),
                    (8, 2, "references"), (9, 4, "calls"), (11, 10, "calls"),
                    (12, 11, "contains"), (13, 11, "implements"), (14, 6, "calls"),
                    (16, 7, "calls"), (17, 16, "calls"), (18, 12, "references"),
                ]
            ]
        }, indent=2),
    }

    # --- 2. Natural language (Wikipedia-style prose) ---
    tests["natural_lang"] = {
        "label": "Natural language (Wikipedia)",
        "text": """The Haber process, also called the Haber-Bosch process, is an artificial nitrogen fixation process and is the main industrial procedure for the production of ammonia today. It is named after its inventors, the German chemists Fritz Haber and Carl Bosch, who developed it in the first decade of the twentieth century. The process converts atmospheric nitrogen to ammonia by a reaction with hydrogen using a metal catalyst under high temperatures and pressures. Before the development of the Haber process, ammonia had been difficult to produce on an industrial scale, with previous techniques being uneconomical. The Haber process is considered one of the most important industrial chemical reactions ever developed, as it enabled the large-scale synthesis of fertilizers and explosives.""",
    }

    # --- 3. Code (Python, Go, TypeScript) ---
    tests["code_python"] = {
        "label": "Code (Python)",
        "text": '''def validate_request(request, schema):
    """Validate incoming API request against schema."""
    errors = []
    for field, rules in schema.items():
        value = request.get(field)
        if rules.get("required") and value is None:
            errors.append(f"Missing required field: {field}")
            continue
        if value is not None and "type" in rules:
            expected = rules["type"]
            if not isinstance(value, expected):
                errors.append(f"Field {field}: expected {expected.__name__}, got {type(value).__name__}")
        if value is not None and "max_length" in rules:
            if len(str(value)) > rules["max_length"]:
                errors.append(f"Field {field}: exceeds max length {rules['max_length']}")
    return {"valid": len(errors) == 0, "errors": errors}


class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = {}

    def is_allowed(self, client_id):
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []
        self.requests[client_id] = [
            t for t in self.requests[client_id] if now - t < self.window
        ]
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        self.requests[client_id].append(now)
        return True''',
    }

    tests["code_go"] = {
        "label": "Code (Go)",
        "text": '''package auth

import (
\t"context"
\t"crypto/rand"
\t"encoding/hex"
\t"errors"
\t"sync"
\t"time"
)

type Session struct {
\tID        string
\tUserID    int64
\tCreatedAt time.Time
\tExpiresAt time.Time
}

type SessionStore struct {
\tmu       sync.RWMutex
\tsessions map[string]*Session
\tttl      time.Duration
}

func NewSessionStore(ttl time.Duration) *SessionStore {
\treturn &SessionStore{
\t\tsessions: make(map[string]*Session),
\t\tttl:      ttl,
\t}
}

func (s *SessionStore) Create(ctx context.Context, userID int64) (*Session, error) {
\tbytes := make([]byte, 32)
\tif _, err := rand.Read(bytes); err != nil {
\t\treturn nil, err
\t}
\tnow := time.Now()
\tsession := &Session{
\t\tID:        hex.EncodeToString(bytes),
\t\tUserID:    userID,
\t\tCreatedAt: now,
\t\tExpiresAt: now.Add(s.ttl),
\t}
\ts.mu.Lock()
\ts.sessions[session.ID] = session
\ts.mu.Unlock()
\treturn session, nil
}

func (s *SessionStore) Get(ctx context.Context, id string) (*Session, error) {
\ts.mu.RLock()
\tdefer s.mu.RUnlock()
\tsession, ok := s.sessions[id]
\tif !ok {
\t\treturn nil, errors.New("session not found")
\t}
\tif time.Now().After(session.ExpiresAt) {
\t\treturn nil, errors.New("session expired")
\t}
\treturn session, nil
}''',
    }

    tests["code_typescript"] = {
        "label": "Code (TypeScript)",
        "text": '''interface CacheEntry<T> {
  value: T;
  expiresAt: number;
  hits: number;
}

class LRUCache<T> {
  private cache = new Map<string, CacheEntry<T>>();
  private readonly maxSize: number;
  private readonly ttlMs: number;

  constructor(maxSize: number = 1000, ttlMs: number = 60000) {
    this.maxSize = maxSize;
    this.ttlMs = ttlMs;
  }

  get(key: string): T | undefined {
    const entry = this.cache.get(key);
    if (!entry) return undefined;
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return undefined;
    }
    entry.hits++;
    this.cache.delete(key);
    this.cache.set(key, entry);
    return entry.value;
  }

  set(key: string, value: T): void {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) this.cache.delete(firstKey);
    }
    this.cache.set(key, {
      value,
      expiresAt: Date.now() + this.ttlMs,
      hits: 0,
    });
  }

  stats(): { size: number; hitRate: number } {
    const entries = Array.from(this.cache.values());
    const totalHits = entries.reduce((sum, e) => sum + e.hits, 0);
    return { size: this.cache.size, hitRate: totalHits / Math.max(entries.length, 1) };
  }
}''',
    }

    # --- 4. Different tabular schemas ---
    random.seed(77777)

    # Users schema
    users = []
    for _ in range(20):
        users.append({
            "userId": f"USR-{random.randint(10000,99999)}",
            "email": f"{random.choice(['alice','bob','carla','dave','eva'])}_{random.randint(1,999)}@example.com",
            "role": random.choice(["admin", "editor", "viewer", "owner"]),
            "active": random.choice([True, False]),
            "loginCount": random.randint(0, 500),
        })

    tests["schema_users"] = {
        "label": "Tabular: users (20 records, 5 fields)",
        "gcf": "## users [20]{userId,email,role,active,loginCount}\n" + "\n".join(
            f"{u['userId']}|{u['email']}|{u['role']}|{str(u['active']).lower()}|{u['loginCount']}" for u in users
        ),
        "json": json.dumps(users, indent=2),
    }

    # Log entries schema
    logs = []
    for _ in range(20):
        logs.append({
            "timestamp": f"2026-06-{random.randint(1,28):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}Z",
            "level": random.choice(["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]),
            "service": random.choice(["api-gateway", "auth-service", "payment-processor", "notification-worker", "search-indexer"]),
            "message": random.choice([
                "Request processed successfully",
                "Connection pool exhausted",
                "Rate limit exceeded for client",
                "Database query timeout after 30s",
                "Cache miss for key",
                "Health check passed",
                "TLS handshake failed",
                "Message queue backlog growing",
            ]),
            "latencyMs": random.randint(1, 5000),
        })

    tests["schema_logs"] = {
        "label": "Tabular: logs (20 records, 5 fields)",
        "gcf": "## logs [20]{timestamp,level,service,message,latencyMs}\n" + "\n".join(
            f"{l['timestamp']}|{l['level']}|{l['service']}|{l['message']}|{l['latencyMs']}" for l in logs
        ),
        "json": json.dumps(logs, indent=2),
    }

    # API response schema (nested)
    api_items = []
    for _ in range(15):
        api_items.append({
            "id": f"item-{random.randint(1000,9999)}",
            "name": random.choice(["Widget A", "Gadget Pro", "Sensor XL", "Module R2", "Adapter Mini"]),
            "price": round(random.uniform(5, 500), 2),
            "currency": random.choice(["USD", "EUR", "GBP"]),
            "available": random.choice([True, False]),
            "tags": random.sample(["electronics", "sale", "new", "popular", "limited", "premium"], k=random.randint(1, 3)),
        })

    tests["schema_api_response"] = {
        "label": "Tabular: API response (15 items, nested tags)",
        "json": json.dumps({"status": "ok", "count": len(api_items), "items": api_items}, indent=2),
        "gcf": f"## items [{len(api_items)}]{{id,name,price,currency,available,tags}}\n" + "\n".join(
            f"{it['id']}|{it['name']}|{it['price']}|{it['currency']}|{str(it['available']).lower()}|{','.join(it['tags'])}" for it in api_items
        ),
    }

    # --- 5. YAML ---
    yaml_users = "users:\n" + "\n".join(
        f"  - userId: {u['userId']}\n    email: {u['email']}\n    role: {u['role']}\n    active: {str(u['active']).lower()}\n    loginCount: {u['loginCount']}"
        for u in users[:10]
    )
    tests["yaml_users"] = {
        "label": "YAML: users (10 records)",
        "text": yaml_users,
    }

    # --- 6. CSV ---
    csv_users = "userId,email,role,active,loginCount\n" + "\n".join(
        f"{u['userId']},{u['email']},{u['role']},{str(u['active']).lower()},{u['loginCount']}" for u in users[:10]
    )
    tests["csv_users"] = {
        "label": "CSV: users (10 records)",
        "text": csv_users,
    }

    return tests


def run_extended_evals(model_a, tok_a, name_a, model_b, tok_b, name_b,
                       device, all_results):
    """Run extended evals across graph, natural language, code, YAML, CSV."""
    tests = get_extended_test_data()

    print()
    print("=" * 80)
    print("EXTENDED EVALUATION")
    print("=" * 80)

    extended_results_a = {}
    extended_results_b = {}

    # Swap model A back to GPU
    model_a.to(device)

    # Categories with GCF vs JSON comparison
    format_tests = {k: v for k, v in tests.items() if "gcf" in v and "json" in v}
    # Categories with just text (natural language, code, YAML, CSV)
    text_tests = {k: v for k, v in tests.items() if "text" in v}

    # --- Format comparison tests (GCF vs JSON) ---
    print()
    print("--- GCF vs JSON by data type ---")
    print()

    if model_b is not None:
        print(f"{'Test':<45} {name_a+' GCF':>12} {name_a+' JSON':>12} {name_b+' GCF':>12} {name_b+' JSON':>12}")
        print("-" * 100)
    else:
        print(f"{'Test':<45} {'GCF PPL':>12} {'JSON PPL':>12} {'Winner':>10}")
        print("-" * 85)

    for key, data in format_tests.items():
        gcf_ppl_a = compute_perplexity(model_a, tok_a, data["gcf"], device)
        json_ppl_a = compute_perplexity(model_a, tok_a, data["json"], device)
        gcf_acc_a, _ = compute_next_token_accuracy(model_a, tok_a, data["gcf"], device)
        json_acc_a, _ = compute_next_token_accuracy(model_a, tok_a, data["json"], device)

        extended_results_a[key] = {
            "label": data["label"],
            "gcf_ppl": gcf_ppl_a, "json_ppl": json_ppl_a,
            "gcf_acc": gcf_acc_a, "json_acc": json_acc_a,
            "gcf_tokens": len(tok_a.encode(data["gcf"]).ids),
            "json_tokens": len(tok_a.encode(data["json"]).ids),
        }

        if model_b is not None:
            # Swap models
            model_a.cpu()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            model_b.to(device)

            gcf_ppl_b = compute_perplexity(model_b, tok_b, data["gcf"], device)
            json_ppl_b = compute_perplexity(model_b, tok_b, data["json"], device)
            gcf_acc_b, _ = compute_next_token_accuracy(model_b, tok_b, data["gcf"], device)
            json_acc_b, _ = compute_next_token_accuracy(model_b, tok_b, data["json"], device)

            extended_results_b[key] = {
                "label": data["label"],
                "gcf_ppl": gcf_ppl_b, "json_ppl": json_ppl_b,
                "gcf_acc": gcf_acc_b, "json_acc": json_acc_b,
                "gcf_tokens": len(tok_b.encode(data["gcf"]).ids),
                "json_tokens": len(tok_b.encode(data["json"]).ids),
            }

            print(f"{data['label']:<45} {gcf_ppl_a:>12.1f} {json_ppl_a:>12.1f} {gcf_ppl_b:>12.1f} {json_ppl_b:>12.1f}")

            # Swap back
            model_b.cpu()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            model_a.to(device)
        else:
            winner = "GCF" if gcf_ppl_a < json_ppl_a else "JSON"
            print(f"{data['label']:<45} {gcf_ppl_a:>12.1f} {json_ppl_a:>12.1f} {winner:>10}")

    # --- Text-only tests (natural language, code, YAML, CSV) ---
    print()
    print("--- Single-format comprehension ---")
    print()

    if model_b is not None:
        print(f"{'Test':<45} {name_a+' PPL':>15} {name_a+' Acc':>10} {name_b+' PPL':>15} {name_b+' Acc':>10} {'Winner':>10}")
        print("-" * 110)
    else:
        print(f"{'Test':<45} {'PPL':>15} {'Accuracy':>10}")
        print("-" * 75)

    for key, data in text_tests.items():
        ppl_a = compute_perplexity(model_a, tok_a, data["text"], device)
        acc_a, _ = compute_next_token_accuracy(model_a, tok_a, data["text"], device)

        extended_results_a[key] = {
            "label": data["label"],
            "ppl": ppl_a, "acc": acc_a,
            "tokens": len(tok_a.encode(data["text"]).ids),
        }

        if model_b is not None:
            model_a.cpu()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            model_b.to(device)

            ppl_b = compute_perplexity(model_b, tok_b, data["text"], device)
            acc_b, _ = compute_next_token_accuracy(model_b, tok_b, data["text"], device)

            extended_results_b[key] = {
                "label": data["label"],
                "ppl": ppl_b, "acc": acc_b,
                "tokens": len(tok_b.encode(data["text"]).ids),
            }

            winner = name_a if ppl_a < ppl_b else name_b
            print(f"{data['label']:<45} {ppl_a:>15.1f} {acc_a:>9.1%} {ppl_b:>15.1f} {acc_b:>9.1%} {winner:>10}")

            model_b.cpu()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            model_a.to(device)
        else:
            print(f"{data['label']:<45} {ppl_a:>15.1f} {acc_a:>9.1%}")

    # Summary
    if model_b is not None:
        print()
        print("--- Extended eval summary ---")
        print()
        gcf_wins = sum(1 for k in format_tests if extended_results_a[k]["gcf_ppl"] < extended_results_b[k]["gcf_ppl"])
        print(f"GCF PPL wins (format tests): {name_a} {gcf_wins}/{len(format_tests)}, {name_b} {len(format_tests)-gcf_wins}/{len(format_tests)}")

        text_wins = sum(1 for k in text_tests if extended_results_a[k]["ppl"] < extended_results_b[k]["ppl"])
        print(f"Text PPL wins: {name_a} {text_wins}/{len(text_tests)}, {name_b} {len(text_tests)-text_wins}/{len(text_tests)}")

    all_results["extended_a"] = extended_results_a
    if model_b is not None:
        all_results["extended_b"] = extended_results_b


def main():
    parser = argparse.ArgumentParser(description="Evaluate structok model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint.pt")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer JSON")
    parser.add_argument("--size", choices=list(MODEL_CONFIGS.keys()), default="410m")
    parser.add_argument("--compare-checkpoint", type=str, default=None,
                        help="Path to second model checkpoint for controlled comparison")
    parser.add_argument("--compare-tokenizer", type=str, default=None,
                        help="Path to second model tokenizer JSON")
    parser.add_argument("--compare-pythia", action="store_true",
                        help="Also evaluate Pythia-410M for comparison")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Path to directory with held-out test files (test-N-records.{json,gcf})")
    parser.add_argument("--extended", action="store_true",
                        help="Run extended evals (graph, natural language, code, YAML, CSV, multi-schema)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (small payloads only)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write results to JSON file")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load primary model
    model, tok, step = load_structok_model(args.checkpoint, args.size, args.tokenizer)
    model.to(device)

    # Load test data (held-out files or generated)
    if args.test_data:
        print(f"Loading held-out test data from {args.test_data}")
        test_data = load_test_files(args.test_data)
        if args.quick:
            test_data = [t for t in test_data if t[0] <= 20]
    else:
        sizes = [5, 10, 20] if args.quick else [5, 10, 20, 50, 100]
        test_data = []
        for n in sizes:
            _, json_str, gcf_str = generate_test_data(n_records=n)
            test_data.append((n, json_str, gcf_str))

    # Eval primary model
    tok_name = Path(args.tokenizer).stem
    print()
    print("=" * 80)
    print(f"Model A: {tok_name} (step {step}) on {device}")
    print("=" * 80)
    results_a = eval_model_on_data(model, tok, test_data, device, tok_name)

    # Compute averages for primary model
    avg_json_ppl_a = sum(r["json_ppl"] for r in results_a) / len(results_a)
    avg_gcf_ppl_a = sum(r["gcf_ppl"] for r in results_a) / len(results_a)
    avg_json_acc_a = sum(r["json_acc"] for r in results_a) / len(results_a)
    avg_gcf_acc_a = sum(r["gcf_acc"] for r in results_a) / len(results_a)
    print()
    print(f"  Avg JSON PPL: {avg_json_ppl_a:.1f}  |  Avg GCF PPL: {avg_gcf_ppl_a:.1f}  |  Ratio: {avg_json_ppl_a / avg_gcf_ppl_a:.2f}x")
    print(f"  Avg JSON Acc: {avg_json_acc_a:.1%}  |  Avg GCF Acc: {avg_gcf_acc_a:.1%}")

    all_results = {"model_a": {"tokenizer": tok_name, "step": step, "results": results_a}}

    # Controlled comparison with second model
    if args.compare_checkpoint and args.compare_tokenizer:
        # Free GPU memory from first model
        model.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        model_b, tok_b, step_b = load_structok_model(
            args.compare_checkpoint, args.size, args.compare_tokenizer)
        model_b.to(device)

        tok_b_name = Path(args.compare_tokenizer).stem
        print()
        print("=" * 80)
        print(f"Model B: {tok_b_name} (step {step_b}) on {device}")
        print("=" * 80)
        results_b = eval_model_on_data(model_b, tok_b, test_data, device, tok_b_name)

        avg_json_ppl_b = sum(r["json_ppl"] for r in results_b) / len(results_b)
        avg_gcf_ppl_b = sum(r["gcf_ppl"] for r in results_b) / len(results_b)
        avg_json_acc_b = sum(r["json_acc"] for r in results_b) / len(results_b)
        avg_gcf_acc_b = sum(r["gcf_acc"] for r in results_b) / len(results_b)
        print()
        print(f"  Avg JSON PPL: {avg_json_ppl_b:.1f}  |  Avg GCF PPL: {avg_gcf_ppl_b:.1f}  |  Ratio: {avg_json_ppl_b / avg_gcf_ppl_b:.2f}x")
        print(f"  Avg JSON Acc: {avg_json_acc_b:.1%}  |  Avg GCF Acc: {avg_gcf_acc_b:.1%}")

        all_results["model_b"] = {"tokenizer": tok_b_name, "step": step_b, "results": results_b}

        # Side-by-side comparison
        print()
        print("=" * 80)
        print("CONTROLLED COMPARISON")
        print("=" * 80)
        print()
        print(f"{'Metric':<25} {tok_name:>18} {tok_b_name:>18} {'Delta':>10}")
        print("-" * 75)
        print(f"{'Avg JSON PPL':<25} {avg_json_ppl_a:>18.1f} {avg_json_ppl_b:>18.1f} {avg_json_ppl_a - avg_json_ppl_b:>+10.1f}")
        print(f"{'Avg GCF PPL':<25} {avg_gcf_ppl_a:>18.1f} {avg_gcf_ppl_b:>18.1f} {avg_gcf_ppl_a - avg_gcf_ppl_b:>+10.1f}")
        print(f"{'JSON/GCF PPL ratio':<25} {avg_json_ppl_a/avg_gcf_ppl_a:>18.2f}x {avg_json_ppl_b/avg_gcf_ppl_b:>18.2f}x")
        print(f"{'Avg JSON Accuracy':<25} {avg_json_acc_a:>17.1%} {avg_json_acc_b:>17.1%} {avg_json_acc_a - avg_json_acc_b:>+10.1%}")
        print(f"{'Avg GCF Accuracy':<25} {avg_gcf_acc_a:>17.1%} {avg_gcf_acc_b:>17.1%} {avg_gcf_acc_a - avg_gcf_acc_b:>+10.1%}")

        print()
        print("Per-size comparison (GCF PPL):")
        print(f"{'Records':<10} {tok_name:>15} {tok_b_name:>15} {'Winner':>10}")
        print("-" * 55)
        a_wins = 0
        for ra, rb in zip(results_a, results_b):
            winner = tok_name if ra["gcf_ppl"] < rb["gcf_ppl"] else tok_b_name
            if ra["gcf_ppl"] < rb["gcf_ppl"]:
                a_wins += 1
            print(f"{ra['records']:<10} {ra['gcf_ppl']:>15.1f} {rb['gcf_ppl']:>15.1f} {winner:>10}")
        print()
        print(f"GCF PPL wins: {tok_name} {a_wins}/{len(results_a)}, {tok_b_name} {len(results_a)-a_wins}/{len(results_a)}")

        # Extended evals (both models on additional formats and data types)
        if args.extended:
            run_extended_evals(model, tok, tok_name, model_b, tok_b, tok_b_name,
                              device, all_results)

        del model_b
        model_b = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    elif args.extended:
        # Extended evals with single model only
        run_extended_evals(model, tok, tok_name, None, None, None,
                          device, all_results)

    # Pythia comparison
    if args.compare_pythia:
        # Ensure model A is off GPU
        model.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        pythia_model, pythia_tok = load_pythia_model()
        pythia_model.to(device)

        class PythiaTokenizerWrapper:
            def __init__(self, hf_tokenizer):
                self.hf_tokenizer = hf_tokenizer
            def encode(self, text):
                class Result:
                    def __init__(self, ids):
                        self.ids = ids
                return Result(self.hf_tokenizer.encode(text))

        pythia_wrapper = PythiaTokenizerWrapper(pythia_tok)

        print()
        print("=" * 80)
        print("Pythia-410M (standard BPE baseline)")
        print("=" * 80)
        results_pythia = eval_model_on_data(pythia_model, pythia_wrapper, test_data, device, "pythia-410m")
        all_results["pythia"] = {"tokenizer": "pythia", "results": results_pythia}

    # Write results to JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {args.output}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
