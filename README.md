# Fake News Detection using Graph RAG

A beginner-friendly project that detects whether a news claim is **FAKE** or **REAL** by
storing thousands of already fact-checked claims in a **graph database**, finding the most
related ones for a new claim, and asking a **Large Language Model (LLM)** to decide — with an
explanation.

This project is inspired by the paper in this folder: [ire_paper.pdf](ire_paper.pdf)
(*GraphRAG-Causal: A Novel Graph-Augmented Framework for Causal Reasoning and Annotation in
News*, arXiv:2506.11600).

---

## Table of Contents

1. [The Idea in One Sentence](#1-the-idea-in-one-sentence)
2. [Key Words Explained](#2-key-words-explained)
3. [What the Reference Paper Did](#3-what-the-reference-paper-did)
4. [What We Do Differently (Our Contribution)](#4-what-we-do-differently-our-contribution)
5. [Problem Statement](#5-problem-statement)
6. [Dataset: LIAR2](#6-dataset-liar2)
7. [Graph Design: Nodes and Edges](#7-graph-design-nodes-and-edges)
8. [Preprocessing Steps](#8-preprocessing-steps)
9. [Building the Graph in Neo4j](#9-building-the-graph-in-neo4j)
10. [Retrieval: The Scoring Formula (with Example)](#10-retrieval-the-scoring-formula-with-example)
11. [Asking the LLM](#11-asking-the-llm)
12. [Evaluation](#12-evaluation)
13. [Optional Extra: Cause and Effect](#13-optional-extra-cause-and-effect)
14. [Design Decisions (FAQ)](#14-design-decisions-faq)
15. [After Prediction: Growing the Knowledge Graph](#15-after-prediction-growing-the-knowledge-graph)
16. [Making Search Fast](#16-making-search-fast)
17. [Algorithms and Models Explained](#17-algorithms-and-models-explained)
18. [Project Folder Structure](#18-project-folder-structure)
19. [Setup and Installation](#19-setup-and-installation)
20. [Timeline](#20-timeline)
21. [What We Are NOT Doing (Keeping It Simple)](#21-what-we-are-not-doing-keeping-it-simple)
22. [References and Links](#22-references-and-links)

---

## 1. The Idea in One Sentence

> We store ~18,000 already fact-checked claims in a **connected graph**. When a **new claim**
> arrives, we use the graph to find the most related old claims and their evidence, give them
> to an **LLM**, and the LLM answers **FAKE or REAL** and explains why.

```
   New claim ──► Find related claims in the graph ──► Give them to the LLM ──► FAKE / REAL + reason
```

---

## 2. Key Words Explained

| Word | Simple meaning |
|---|---|
| **Graph** | A network of dots connected by lines (like a map of friendships). |
| **Node** | One dot. Here: a claim, a person, a topic, a place, etc. |
| **Edge / Relationship** | One line connecting two dots, e.g. *Chris Abele* **SAID** *this claim*. |
| **Neo4j** | A free database designed to store and search graphs. |
| **Cypher** | The query language for Neo4j (like SQL, but for graphs). |
| **Embedding** | A list of numbers representing a sentence's meaning. Similar meanings → similar numbers. |
| **Cosine similarity** | A number between 0 and 1 telling how similar two embeddings are (1 = same meaning). |
| **NER (Named Entity Recognition)** | A tool that finds names of people, places, organisations in text. |
| **Entity** | A named thing found by NER, e.g. *Wisconsin*, *Scott Walker*. |
| **LLM** | Large Language Model (e.g. Llama, GPT) — an AI that reads and writes text. |
| **RAG** | Retrieval-Augmented Generation: *look up helpful information first, then let the LLM answer*. |
| **Graph RAG** | RAG where the "look up" step searches a **graph** instead of plain text. |
| **Few-shot examples** | A few solved examples shown to the LLM so it understands the task. |

---

## 3. What the Reference Paper Did

The paper **GraphRAG-Causal** works in three stages:

1. **Data preparation** — about 1,000 news sentences (from the *Causal News Corpus*, about
   protests) were manually tagged with a **cause**, an **effect** and a **trigger word**
   (e.g. *because*, *due to*). Each sentence became a tiny graph:
   `Cause → Event → Effect` and `Event → Trigger`.
2. **Graph retrieval** — the tiny graphs were stored in Neo4j with embeddings. For a new
   sentence, a score picked the most similar stored sentences:
   `H(e) = α · cosine_similarity + β · S(e)`, where `S(e) = 1` if the stored event has any
   cause/effect/trigger, otherwise 0.
3. **LLM inference** — the top-k stored sentences were put into an XML prompt as examples, and
   the LLM answered: *"Is this sentence causal (1) or not (0)?"* and tagged the cause/effect.

**Result:** best F1 ≈ 0.83, roughly equal to (slightly below) a fine-tuned model baseline.

### Problems we noticed in the paper

| Problem | Simple explanation |
|---|---|
| The graphs are **not connected** | Every sentence is its own little island. No two sentences share a node, so the "graph" adds almost nothing beyond a normal text search. |
| The structural score `S(e)` is almost useless | Almost every stored sentence is causal, so almost every one gets `S(e) = 1`. It gives the same bonus to everyone, so the ranking is really just text similarity. It also never looks at the *new* sentence. |
| Example bias | Retrieved examples are mostly "causal" ones, so the LLM leans toward answering "causal" (high recall ~0.93, lower precision ~0.72 for one model). |
| **No fake-news detection** | The paper mentions misinformation as a use case but never checks whether news is *true or false*. |
| Tagging not evaluated | The cause/effect tags were never measured against a gold standard. |

---

## 4. What We Do Differently (Our Contribution)

| | Paper | Our project |
|---|---|---|
| **Task** | Is the sentence causal? | **Is the claim FAKE or REAL?** (+ explanation) |
| **Dataset** | Causal News Corpus (~1,000 sentences) | **LIAR2** (~23,000 fact-checked claims) |
| **Graph** | Separate tiny islands | **One connected graph** — claims linked through shared speakers, subjects, entities |
| **Score** | Text similarity + a bonus nearly everyone gets | Text similarity + **real graph connections to the new claim** |
| **Examples for LLM** | Mostly one class | **Balanced**: equal FAKE and REAL examples |
| **Evidence** | None | Journalist **evidence sentences** attached to each past claim |

---

## 5. Problem Statement

**Given** a new public claim (for example, something a politician said), along with *who* said
it, *where* they said it, and *what topic* it is about,

**predict** whether the claim is **FAKE** or **REAL**, and **explain** the decision using
evidence from previously fact-checked claims,

**by** building a connected knowledge graph of past fact-checked claims and retrieving the most
related claims and evidence from it (Graph RAG) to guide an LLM.

**Research question:** *Does retrieving examples through a connected graph give the LLM better
help than retrieving examples by text similarity alone?*

---

## 6. Dataset: LIAR2

- **Link:** <https://huggingface.co/datasets/chengxuphd/liar2>
- **GitHub:** <https://github.com/chengxuphd/liar2>
- **Size:** ~23,000 claims → train ~18,400 · validation ~2,300 · test ~2,300 (already split)
- **Source:** Claims by US public figures, checked by journalists at **PolitiFact**.
- **License:** Apache-2.0

**We only need this one dataset.**

### Columns

| Column | Meaning | Used for |
|---|---|---|
| `id` | Unique number of the claim | Claim node ID |
| `label` | Verdict (0–5, see below) | The answer we predict |
| `statement` | The claim text | Claim node text + embedding |
| `date` | When the claim was made | Claim node property |
| `subject` | Topics, separated by `;` | Subject nodes |
| `speaker` | Who said it | Speaker node |
| `speaker_description` | Short bio of the speaker | Speaker property, party |
| `state_info` | Speaker's US state | State node |
| `true_counts` … `pants_on_fire_counts` | How often this speaker got each verdict | Speaker's track record (optional) |
| `context` | Where it was said (tweet, speech, TV ad…) | Context node |
| `justification` | Journalist's explanation of the verdict | Evidence nodes (training claims only!) |

### Labels

| `label` | Meaning | Our binary label |
|---|---|---|
| 0 | Pants on fire | **FAKE** |
| 1 | False | **FAKE** |
| 2 | Barely true | **FAKE** |
| 3 | Half true | **REAL** |
| 4 | Mostly true | **REAL** |
| 5 | True | **REAL** |

> Check the mapping by reading a few rows and their justifications after loading. Mention in
> the report that *half-true* was counted as REAL (a common choice).

### A real example row

| Column | Value |
|---|---|
| id | 13847 |
| label | 5 (True → REAL) |
| statement | 90 percent of Americans "support universal background checks" for gun purchases. |
| date | October 2, 2017 |
| subject | government regulation;polls and public opinion;guns |
| speaker | chris abele |
| speaker_description | Chris Abele is Milwaukee County Executive … Abele has indicated he is a Democrat. |
| state_info | wisconsin |
| context | a tweet |
| justification | "Universal" is the term for background checks to be done on every gun sale. We found support for that policy at 94 percent in the latest national poll. Support ranged between 84 percent and 89 percent in the four other most recent polls. Experts say support at or near 90 percent has been consistent for years. … |

---

## 7. Graph Design: Nodes and Edges

### Node types (the dots)

| Node | Represents | Comes from | Properties |
|---|---|---|---|
| **Claim** | One fact-checked sentence | `statement`, `label`, `date` | id, text, label, date, embedding |
| **Speaker** | The person who said it | `speaker`, `speaker_description` | name, description |
| **Party** | Political party | parsed from `speaker_description` | name |
| **State** | US state | `state_info` | name |
| **Subject** | A topic | `subject` split on `;` | name |
| **Context** | Where it was said | `context` | name |
| **Entity** | A named thing inside the claim | spaCy NER on `statement` | name, type |
| **Evidence** | One sentence of the journalist's explanation | `justification` split into sentences | text, embedding |

### Edge types (the lines)

```
(Speaker)-[:SAID]->(Claim)
(Speaker)-[:MEMBER_OF]->(Party)
(Speaker)-[:FROM]->(State)
(Claim)-[:ABOUT]->(Subject)
(Claim)-[:SAID_IN]->(Context)
(Claim)-[:MENTIONS]->(Entity)
(Claim)-[:HAS_EVIDENCE]->(Evidence)
```

### The example row as a graph

```
           [Party: Democrat]          [State: Wisconsin]
                  ▲                          ▲
              MEMBER_OF                    FROM
                  │                          │
             [Speaker: Chris Abele]──────────┘
                        │
                       SAID
                        ▼
 [Context: a tweet] ◄─SAID_IN─ [CLAIM 13847: "90 percent of Americans support
                                universal background checks…"  label = REAL]
        ┌──────────────┬───────────────┼────────────────┬──────────────────┐
      ABOUT          ABOUT           ABOUT          MENTIONS        HAS_EVIDENCE
        ▼              ▼               ▼                ▼                  ▼
 [Subject: guns] [Subject: polls  [Subject: govt   [Entity:        [Evidence: "We found support
                  and public       regulation]      Americans]      … at 94 percent in the latest
                  opinion]                                          national poll."]
```

### Why the graph is *connected* (the most important idea)

We load **all ~18,400 training rows**, and we **never create the same node twice**
(using Cypher's `MERGE`). So:

- all claims by Chris Abele connect to **one** `Speaker: chris abele` node,
- all gun-related claims, from every speaker, connect to **one** `Subject: guns` node,
- all claims mentioning Wisconsin connect to **one** `Entity: Wisconsin` node.

The individual rows join into **one big network**:

```
 [Claim X: "Gun deaths fell…" FAKE] ──ABOUT──┐
                                             ▼
 [Claim 13847  REAL] ────────ABOUT─────► [Subject: guns] ◄──ABOUT── [Claim Y: "…" REAL]
        ▲                                                                   │
       SAID                                                             MENTIONS
        │                                                                   ▼
 [Speaker: Chris Abele] ──SAID──► [Claim Z: another Abele claim]    [Entity: Wisconsin]
```

When a new claim about guns or Wisconsin arrives, we can **walk** the graph to related
claims and their evidence. The paper's graph could not do this because its pieces never touched.

---

## 8. Preprocessing Steps

### Step 1 — Load the data

```python
from datasets import load_dataset

ds = load_dataset("chengxuphd/liar2")
train = ds["train"].to_pandas()
val   = ds["validation"].to_pandas()
test  = ds["test"].to_pandas()
print(train.columns)
```

### Step 2 — Create the binary label

```python
for df in (train, val, test):
    df["binary"] = df["label"].apply(lambda x: "FAKE" if x <= 2 else "REAL")
```

### Step 3 — Clean the text

- Strip extra spaces and line breaks (`\r\n`).
- Remove the leading word **"Says"** that PolitiFact adds to many statements.
- Lowercase and strip `speaker`, `state_info`, `context` so the same person/place always
  becomes the same node.
- Drop rows with an empty `statement`.

```python
import re

def clean_statement(s):
    s = re.sub(r"\s+", " ", str(s)).strip()
    s = re.sub(r"^(Says|Said)( that)?\s+", "", s)
    return s

for df in (train, val, test):
    df["statement"] = df["statement"].apply(clean_statement)
    for col in ("speaker", "state_info", "context"):
        df[col] = df[col].fillna("").str.lower().str.strip()
```

### Step 4 — Split subjects into a list

```python
for df in (train, val, test):
    df["subjects"] = df["subject"].fillna("").apply(
        lambda s: [x.strip().lower() for x in s.split(";") if x.strip()]
    )
```

### Step 5 — Prevent cheating (data leakage) ⚠️

1. **Hide the answer key.** The `justification` explains the verdict, so it is effectively
   the answer. Create Evidence nodes **only for training claims**. Never use a test claim's
   own justification when predicting it.
2. **Fix the speaker history.** The count columns (`false_counts` etc.) may include the
   claim's own verdict. For each row, subtract 1 from the count matching its own label before
   using these counts.

```python
count_cols = ["pants_on_fire_counts", "false_counts", "mostly_false_counts",
              "half_true_counts", "mostly_true_counts", "true_counts"]  # index = label 0..5

def remove_own_label(row):
    col = count_cols[row["label"]]
    row[col] = max(0, row[col] - 1)
    return row

for df in (train, val, test):
    df[count_cols] = df.apply(remove_own_label, axis=1)[count_cols]
```

### Step 6 — Find entities with spaCy

```python
import spacy
nlp = spacy.load("en_core_web_sm")

KEEP = {"PERSON", "ORG", "GPE", "NORP", "LOC", "LAW", "EVENT"}

def get_entities(text):
    return sorted({e.text.lower() for e in nlp(text).ents if e.label_ in KEEP})

for df in (train, val, test):
    df["entities"] = df["statement"].apply(get_entities)
```

### Step 7 — Split evidence into sentences (training only)

```python
train["evidence"] = train["justification"].fillna("").apply(
    lambda t: [s.text.strip() for s in nlp(t).sents if len(s.text.strip()) > 20]
)
```

### Step 8 — Make embeddings

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")   # same model as the paper, 384 numbers

for df in (train, val, test):
    df["emb"] = list(model.encode(df["statement"].tolist(), normalize_embeddings=True))
```

Save the processed data (e.g. `data/processed/train.pkl`) so these steps only run once.

---

## 9. Building the Graph in Neo4j

### Start Neo4j with Docker

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
```

Open <http://localhost:7474> in a browser (user `neo4j`, password `password`) to see the graph.

> Before loading, create the uniqueness constraints from
> [Section 16, Trick 3](#16-making-search-fast) so loading stays fast.

### Load one claim (Cypher)

`MERGE` = *create only if it does not exist yet*. This is what makes the graph connected.

```cypher
MERGE (c:Claim {id: $id})
  SET c.text = $statement, c.label = $binary, c.date = $date, c.embedding = $emb
MERGE (s:Speaker {name: $speaker})
MERGE (s)-[:SAID]->(c)
MERGE (st:State {name: $state})
MERGE (s)-[:FROM]->(st)
MERGE (ctx:Context {name: $context})
MERGE (c)-[:SAID_IN]->(ctx)
WITH c
UNWIND $subjects AS subj
  MERGE (t:Subject {name: subj})
  MERGE (c)-[:ABOUT]->(t)
```

Entities and evidence are added the same way:

```cypher
MATCH (c:Claim {id: $id})
UNWIND $entities AS ent
  MERGE (e:Entity {name: ent})
  MERGE (c)-[:MENTIONS]->(e)
```

```cypher
MATCH (c:Claim {id: $id})
UNWIND range(0, size($evidence) - 1) AS i
  CREATE (ev:Evidence {text: $evidence[i], idx: i})
  CREATE (c)-[:HAS_EVIDENCE]->(ev)
```

### Python loader (sketch)

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    for _, r in train.iterrows():
        session.run(LOAD_CLAIM_QUERY, id=int(r["id"]), statement=r["statement"],
                    binary=r["binary"], date=r["date"], emb=r["emb"].tolist(),
                    speaker=r["speaker"], state=r["state_info"], context=r["context"],
                    subjects=r["subjects"])
        # ...then the entity and evidence queries
```

### Vector index (for fast similarity search)

```cypher
CREATE VECTOR INDEX claim_emb IF NOT EXISTS
FOR (c:Claim) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
```

> Only **training** claims go into the graph. Validation and test claims are the "new claims"
> we try to predict.

---

## 10. Retrieval: The Scoring Formula (with Example)

### The formula

```
score = 0.6 × (text similarity)                          ← 0 to 1
      + 0.2 × (1 if same speaker OR same subject, else 0)
      + 0.2 × (shared entities ÷ all distinct entities)   ← 0 to 1
```

| Part | Question it answers | Where it comes from |
|---|---|---|
| Text similarity | Do the two claims *mean* similar things? | Cosine similarity of embeddings |
| Same speaker / subject | Are they linked through a Speaker or Subject node? | Graph path |
| Shared entities | Do they talk about the same people / places? | Shared Entity nodes |

The weights (0.6 / 0.2 / 0.2) are a starting point. Later, try other values on the
**validation set** and keep the best.

### Worked example (claims made up for teaching)

**New claim:** *"Scott Walker's tax cuts cost Wisconsin 50,000 jobs."*
Subjects: taxes, jobs · Entities: {Scott Walker, Wisconsin}

| | Old claim in graph | Subjects | Entities | Verdict |
|---|---|---|---|---|
| A | "Wisconsin added jobs every month under Scott Walker." | jobs, economy | {Wisconsin, Scott Walker} | REAL |
| B | "Scott Walker's tax cuts cost Wisconsin 40,000 jobs." | taxes | {Scott Walker, Wisconsin} | FAKE |
| C | "Texas lost 50,000 jobs after its tax cuts." | taxes, jobs | {Texas} | FAKE |
| D | "Unemployment in Ohio rose last year." | economy | {Ohio} | REAL |

| | Text sim | Same speaker/subject | Shared entities | Score |
|---|---|---|---|---|
| A | 0.72 | 1 (jobs) | 2/2 = 1.0 | 0.432 + 0.2 + 0.2 = **0.832** |
| B | 0.93 | 1 (taxes) | 2/2 = 1.0 | 0.558 + 0.2 + 0.2 = **0.958** |
| C | 0.80 | 1 (taxes, jobs) | 0/3 = 0.0 | 0.480 + 0.2 + 0.0 = **0.680** |
| D | 0.55 | 0 | 0/3 = 0.0 | 0.330 + 0.0 + 0.0 = **0.330** |

| Rank | Text only (normal RAG) | Our graph score |
|---|---|---|
| 1 | B | B |
| 2 | **C** (Texas — wrong state) | **A** (same person, same state) |
| 3 | A | C |
| 4 | D | D |

Claim C *sounds* similar but is about Texas. Claim A uses different words but is about the
same person and state, and it directly challenges the new claim. **The graph moves the more
useful evidence up.**

### Balanced picking

From the ranked list, take the **top 5 FAKE** and the **top 5 REAL** claims (10 total), so the
LLM always sees both sides. Then collect their Evidence sentences from the graph.

### Retrieval steps in code (sketch)

1. Embed the new claim.
2. Get the 50 most similar training claims from the vector index:

   ```cypher
   CALL db.index.vector.queryNodes('claim_emb', 50, $query_emb) YIELD node AS c, score
   WITH c, 2 * score - 1 AS sim   // Neo4j returns (1 + cosine) / 2 → convert back to plain cosine
   OPTIONAL MATCH (c)<-[:SAID]-(s:Speaker)
   OPTIONAL MATCH (c)-[:ABOUT]->(t:Subject)
   OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
   OPTIONAL MATCH (c)-[:HAS_EVIDENCE]->(ev:Evidence)
   RETURN c.id AS id, c.text AS text, c.label AS label, sim,
          s.name AS speaker, collect(DISTINCT t.name) AS subjects,
          collect(DISTINCT e.name) AS entities, collect(DISTINCT ev.text) AS evidence
   ```

3. In Python, compute the score for each candidate:

   ```python
   def graph_score(cand, query):
       same = (cand["speaker"] == query["speaker"]) or bool(set(cand["subjects"]) & set(query["subjects"]))
       q_ent, c_ent = set(query["entities"]), set(cand["entities"])
       ent = len(q_ent & c_ent) / len(q_ent | c_ent) if (q_ent | c_ent) else 0.0
       return 0.6 * cand["sim"] + 0.2 * int(same) + 0.2 * ent
   ```

4. Sort by score, take the top 5 FAKE + top 5 REAL.

---

## 11. Asking the LLM

### Prompt template

```
You are a professional fact-checker. Decide if the NEW CLAIM is FAKE or REAL.

Here are similar claims that were already fact-checked, with evidence:

1. "Wisconsin added jobs every month under Scott Walker." (speaker: ..., subject: jobs) → REAL
   Evidence: "..."
2. "Scott Walker's tax cuts cost Wisconsin 40,000 jobs." (speaker: ..., subject: taxes) → FAKE
   Evidence: "..."
... (10 examples: 5 FAKE, 5 REAL)

NEW CLAIM: "<statement>"
Speaker: <speaker> | Said in: <context> | Subjects: <subjects>

Think about whether the evidence supports or contradicts the new claim.
Answer ONLY in JSON:
{"label": "FAKE" or "REAL", "reason": "<2-3 sentences citing the examples you used>"}
```

### Which LLM

- **Llama 3.1 8B Instruct** — free API through **Groq** (<https://groq.com>, the service the
  paper used; model name `llama-3.1-8b-instant`), or run it locally with **Ollama**
  (<https://ollama.com>).
- Use `temperature = 0` so answers are repeatable.
- Parse the JSON; if parsing fails, retry once.

---

## 12. Evaluation

### Three systems to compare (this table is the main result)

| System | What the LLM gets |
|---|---|
| **1. LLM alone** | Only the new claim (no examples) |
| **2. Normal RAG** | New claim + 10 most similar claims by **text similarity only** |
| **3. Graph RAG (ours)** | New claim + 10 claims chosen by the **graph score**, balanced, with evidence |

If system 3 beats system 2, the graph is helping — that is the project's main finding.

### Metrics (same as the paper)

| Metric | Meaning |
|---|---|
| **Accuracy** | % of claims labelled correctly |
| **Precision** (FAKE) | Of claims we called FAKE, how many really were FAKE |
| **Recall** (FAKE) | Of the really FAKE claims, how many we caught |
| **F1** | Balance of precision and recall |
| **MCC** | One number for overall quality, fair even if classes are unbalanced |

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
```

> The test set has ~2,300 claims. If LLM calls are slow or limited, use a fixed random sample
> of **500 test claims** (same sample for all three systems).

### Extra experiments (if time allows)

- Number of examples: 4, 10, 20 (the paper tested 5–50).
- Remove one part of the score at a time (no subject/speaker part, no entity part) to see
  which part helps most.
- Remove the evidence sentences to see how much they help.
- Show 5–10 example explanations in the report (good and bad).

---

## 13. Optional Extra: Cause and Effect

This links our project back to the paper. Some claims contain cause and effect, e.g.
*"Obamacare caused premiums to double."*

1. Ask the LLM to extract `{"cause": ..., "effect": ..., "trigger": ...}` from each claim
   (or leave empty if there is none).
2. Add nodes and edges:

   ```
   (Claim)-[:CLAIMS_CAUSE]->(Cause: "obamacare")
   (Claim)-[:CLAIMS_EFFECT]->(Effect: "premiums doubled")
   ```

3. Now claims that blame the same cause for the same effect are connected, and the score can
   get a fourth part: *"does the old claim share the same cause or effect?"*

Do this **only after** the basic system works.

---

## 14. Design Decisions (FAQ)

### Q1. How do we decide which old claims to give the LLM?

**The score does NOT decide fake or real.** It never looks at an old claim's label. It only
measures *"how related is this old claim to the new claim?"* — a **relevance** score. Deciding
FAKE or REAL is the LLM's job at the very end.

The picking process, using the example from [Section 10](#10-retrieval-the-scoring-formula-with-example):

**Step 1 — Score every candidate (labels ignored):**

| Old claim | Score |
|---|---|
| B: "Scott Walker's tax cuts cost Wisconsin 40,000 jobs." | 0.958 |
| A: "Wisconsin added jobs every month under Scott Walker." | 0.832 |
| C: "Texas lost 50,000 jobs after its tax cuts." | 0.680 |
| D: "Unemployment in Ohio rose last year." | 0.330 |

**Step 2 — Only now look at labels and split into two lists:**

| FAKE list (by score) | REAL list (by score) |
|---|---|
| B — 0.958 | A — 0.832 |
| C — 0.680 | D — 0.330 |

**Step 3 — Take the top ones from each list.** With 2 examples → **B (FAKE) and A (REAL)**.
In the real system: 50 candidates → **top 5 FAKE + top 5 REAL**.

**Step 4 — The LLM decides.** It sees B (a similar claim that was FAKE) and A (a REAL claim
saying Wisconsin *added* jobs under Walker). A's evidence contradicts the new claim, so the LLM
should lean towards FAKE.

**Why balanced (5 + 5) and not simply the top 10?** If the top 10 were 9 FAKE and 1 REAL, the
LLM could take a shortcut — *"most similar claims are fake, so this is fake"* — copying the
majority instead of reading the evidence. The reference paper had this problem (its examples were
mostly one class, so the model over-predicted that class). With 5 + 5 the LLM must decide **which
side the new claim really resembles**.

**This is tested, not assumed:** run the system with plain top-10 and with balanced 5 + 5, and
report both F1 scores.

### Q2. Why these 3 factors and these weights? Why not others?

The 3 factors and the weights 0.6 / 0.2 / 0.2 are a **sensible starting point, not a proven
best**. The project's job is to test them.

**Why these 3:** each one captures a *different kind* of relatedness, and every claim has the
information needed to compute it.

| Factor | Question it answers | What it catches that the others miss |
|---|---|---|
| Text similarity | Do they *mean* similar things? | Same idea in different words ("jobs lost" vs "unemployment rose") |
| Same speaker / subject | Same *source* or same *topic*? | Claims on the same topic written very differently |
| Shared entities | Same *specific* people / places? | Wisconsin vs Texas — text similarity mixes these up |

**Why text similarity gets the biggest weight (0.6):** it is the strongest single signal, it
works for every claim (some claims have no entities), and it is what the reference paper relied
on. The other two factors *adjust* the ranking rather than replace it.

**How to find better weights — grid search on the validation set.** Try several combinations,
run on the **validation** set (never the test set), keep the best. Illustrative example:

| Text | Speaker/Subject | Entities | Validation F1 |
|---|---|---|---|
| 1.0 | 0.0 | 0.0 | 0.70 ← this equals "normal RAG" |
| 0.6 | 0.2 | 0.2 | 0.74 |
| 0.5 | 0.25 | 0.25 | 0.73 |
| 0.7 | 0.1 | 0.2 | 0.75 ← best → use on test set |

Tuning on the test set would be tuning on the answers, and the final numbers would look better
than they really are.

**Other factors we considered:**

| Possible factor | Add it? | Why |
|---|---|---|
| Date closeness (claims made around the same time) | ✅ Good to try | News about the same event happens at the same time |
| Same context (both tweets, both TV ads) | ✅ Can try | Cheap; probably a small effect |
| Same cause/effect ([Section 13](#13-optional-extra-cause-and-effect)) | ✅ Good to try | Links our work to the reference paper |
| Same state | 🟡 Maybe | Helps local claims, adds noise for national ones |
| Speaker's past record ("this person often lies") | ⚠️ Careful | The model starts judging the **person**, not the **claim** |
| Same political party | ❌ Avoid | Builds political bias into a fact-checker — ethically bad |

**Why not add everything at once?**

1. More weights → more tuning → the "best" combination may just fit the validation set by luck
   (overfitting).
2. Harder to explain why a claim was picked.
3. We could not tell which factor actually helped.

**Rule we follow:** start with 3 factors; **add one new factor at a time**; keep it **only if
validation F1 improves**.

**Possible improvement:** "same speaker OR same subject" mixes two things. Splitting it into two
separate factors lets us test them individually — and drop the speaker part if it causes the
"judging the person" bias.

### Q3. How do we get the subject of a claim?

**In LIAR2 we don't extract it — it is given.** The `subject` column was filled in by PolitiFact
editors for every claim (train, validation and test). We just split it on `;`:

```
"government regulation;polls and public opinion;guns"
            ↓ split on ";"
["government regulation", "polls and public opinion", "guns"]
```

```python
subjects = [x.strip().lower() for x in row["subject"].split(";") if x.strip()]
```

**Subjects vs entities — don't mix them up:**

| | Subjects | Entities |
|---|---|---|
| Example | "guns", "taxes", "health care" | "Wisconsin", "Scott Walker", "Obamacare" |
| Source | **Given** in the dataset (PolitiFact editors) | **Extracted** by us with spaCy |
| Meaning | A broad topic | A specific named thing in the sentence |

**But a brand-new claim from the internet has no subject column.** For a live demo, three
options (easiest first):

- **Option A — Borrow from neighbours (easiest):** find the 10 most similar training claims and
  use the subjects that appear most often among them. *Example: 7 of 10 neighbours have "guns" →
  the new claim gets "guns".*
- **Option B — Ask the LLM:** give it the list of all LIAR2 subject names and ask it to pick 1–3
  as a JSON list. The fixed list is essential, otherwise the LLM invents topic names that match no
  node in the graph.
- **Option C — Train a small classifier:** TF-IDF + logistic regression on LIAR2 (text →
  subjects), and measure its accuracy on the test set.

**Recommendation:** use the given `subject` column for the **main experiments** (clean and fair —
every system gets the same information). Use **Option A** for a **live demo**. If time allows,
compare results with real vs predicted subjects to show how the system behaves outside the
dataset.

---

## 15. After Prediction: Growing the Knowledge Graph

### Do we add a predicted claim to the graph, or discard it?

**Neither blindly.** During experiments we add **nothing**. In a real system we **store** it,
clearly marked as *predicted, not verified*.

### Why not add every predicted claim as a fact?

Because the LLM is sometimes **wrong**, and a wrong label saved as fact gets copied:

```
Day 1:  LLM wrongly says Claim X is FAKE  → saved in graph as FAKE
Day 2:  New Claim Y (similar to X) arrives → retrieval finds X ("FAKE") as an example
        → LLM is pushed towards FAKE again → wrong again → saved
Day 3:  X and Y now both "prove" the next similar claim is FAKE…
```

The graph slowly fills with the model's own mistakes. This is a **feedback loop**
("self-poisoning"). The graph is valuable only because it contains **human-verified facts**.

### Rule 1 — During experiments: the graph is frozen

The graph contains **training claims only** while we evaluate. If test claim #1 were added, it
could help test claim #2 — results would depend on the order we ran them, and the comparison of
the three systems would not be fair. Every test claim is judged against **the same graph**.

### Rule 2 — In a real system: two kinds of claims

Every Claim node gets extra properties:

| Property | Values | Meaning |
|---|---|---|
| `status` | `verified` / `predicted` | Confirmed by a human fact-checker, or only by our model? |
| `source` | `politifact` / `our_model` / `human_reviewer` | Where the label came from |
| `confidence` | 0 – 1 (predicted claims only) | How sure the LLM was |

Retrieval uses **only verified claims** as examples:

```cypher
CALL db.index.vector.queryNodes('claim_emb', 50, $query_emb) YIELD node AS c, score AS sim
WHERE c.status = 'verified'
...
```

Predicted claims are stored, but **never used as evidence**, so they cannot poison the system.

### The full process after labelling

```
                         New claim arrives
                                │
                                ▼
                 Graph RAG + LLM → "FAKE", confidence 0.85, reason
                                │
                                ▼
          Shown to user  +  saved in graph with status = "predicted"
                                │
                                ▼
          Linked to its Speaker / Subject / Entity nodes (like any claim)
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
   A human fact-checker reviews it      Nobody reviews it
   (or PolitiFact publishes a verdict)          │
                 │                             ▼
                 ▼                   Stays "predicted":
   status → "verified"               NOT used as an example,
   label  → the human's verdict      but still useful (below)
   evidence → the human's explanation
                 │
                 ▼
   Now used as an example for future claims ✅
```

### Why keep predicted claims at all?

1. **Spotting repeated claims.** Fake news repeats. If the same predicted-FAKE claim appears 30
   times this week, it shows up linked to the same Entity / Subject nodes — a sign it is trending
   and should be checked by a human first.
2. **A priority queue for human checkers.** Sort predicted claims by *spreading fast + low
   confidence* so humans check the most important ones first.
3. **No repeated work.** If an identical claim arrives again, reuse the stored answer instead of
   calling the LLM again.

### How the knowledge graph grows (safely)

| Way | How | Safe? |
|---|---|---|
| **New fact-checks** | Periodically download new PolitiFact (or other fact-checker) verdicts and run them through the same pipeline: clean → entities → embed → `MERGE` | ✅ Human-verified |
| **Human review** | Predicted claims confirmed by a human are promoted to `verified` | ✅ Human-verified |
| **More sources** | Later, add other fact-check datasets (e.g. PUBHEALTH for health) into the same graph | ✅ Human-verified |

Adding claims is cheap. Because we use `MERGE`, a new Chris Abele claim attaches to the
**existing** Chris Abele node and a new "guns" claim to the **existing** "guns" node. Nothing
else is rebuilt, and Neo4j updates the vector index automatically.

### Optional experiment: self-training

Test what happens if we **do** add high-confidence predictions (e.g. confidence > 0.9) as
examples:

1. Split the test set into two halves.
2. Predict half 1 and add those predictions to the graph.
3. Predict half 2, and compare its F1 **with** and **without** the added claims.

If F1 drops, we have shown the feedback-loop problem with real numbers. Either outcome is a
useful finding.

---

## 16. Making Search Fast

### Do we search the whole graph for every claim? No.

Work done **once in advance** (preprocessing) lets each search touch only a **tiny piece** of
the graph:

```
Whole graph: ~18,400 claims, ~100,000 evidence sentences, thousands of speakers/subjects/entities
┌───────────────────────────────────────────────────────────────┐
│ · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · │
│ · · · · · · · · ┌───────────────────┐ · · · · · · · · · · · · │
│ · · · · · · · · │  50 candidates +  │ · · · · · · · · · · · · │  ← we only
│ · · · · · · · · │  their direct     │ · · · · · · · · · · · · │    touch
│ · · · · · · · · │  neighbours       │ · · · · · · · · · · · · │    this box
│ · · · · · · · · └───────────────────┘ · · · · · · · · · · · · │
│ · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · │
└───────────────────────────────────────────────────────────────┘
```

### Trick 1 — Do heavy work only once

| Done ONCE, in advance | Done for EACH new claim |
|---|---|
| Embeddings of all 18,400 claims | Embedding of **1** new claim |
| Entities of all claims (spaCy) | Entities of **1** new claim |
| Evidence sentences split | — |
| Graph built in Neo4j | — |
| Indexes built (Tricks 2 and 3) | — |

At search time we only process **one sentence**.

### Trick 2 — Vector index: don't compare with every claim

Without an index, the new claim is compared with all 18,400 claims one by one — like finding a
book by walking past every shelf. Neo4j's vector index (method: **HNSW**) organises all
embeddings in advance into a *map of neighbourhoods* with zoom levels:

```
Level 3 (zoomed out):   a few "landmark" claims   → "you're near the GUNS area"
Level 2:                more claims                → "near GUN POLLS"
Level 1:                more claims                → "near BACKGROUND-CHECK polls"
Level 0 (all claims):   the exact closest 50      → done
```

Like zooming on a map from country → city → street, it looks at only a few hundred claims
instead of all of them, and stays fast even with **millions** of claims. Created once (see
[Section 9](#9-building-the-graph-in-neo4j)):

```cypher
CREATE VECTOR INDEX claim_emb IF NOT EXISTS
FOR (c:Claim) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
```

### Trick 3 — Name indexes: finding a node is instant

While building the graph, `MERGE` constantly asks *"does Speaker 'chris abele' already exist?"*.
Without an index Neo4j scans every Speaker. Uniqueness constraints (which also create indexes)
fix this — run them **before** loading the data:

```cypher
CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim)   REQUIRE c.id   IS UNIQUE;
CREATE CONSTRAINT speaker  IF NOT EXISTS FOR (s:Speaker) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT subject  IF NOT EXISTS FOR (t:Subject) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT entity   IF NOT EXISTS FOR (e:Entity)  REQUIRE e.name IS UNIQUE;
```

### Trick 4 — Walk outward from the 50 candidates, never inward from a big hub

Some nodes are **huge hubs** — the Subject "economy" might connect to 2,000 claims.

```
❌  [Subject: economy] ──► 2,000 claims ──► score all          (fans out from a hub)
✅  50 candidates ──► their own 2–3 subjects each ──► compare   (1 step from each candidate)
```

- ❌ *"Find all claims about economy, then score them"* → touches 2,000 claims, worse as the graph grows.
- ✅ *"For these 50 candidates, is each one about economy?"* → 50 quick checks.

Per search we read 50 candidates plus their speaker, subjects, entities and evidence (one step
away) — **a few hundred nodes** out of ~150,000+.

### Trick 5 (optional, very large graphs) — Filter first

With millions of claims, narrow the area **before** the similarity search, e.g. only claims that
share at least one subject with the new claim, or only claims from the last 5 years.

### Honest note on our project size

LIAR2 has only ~18,400 training claims, so even a plain one-by-one comparison takes well under a
second — speed is **not** a real problem here. These tricks matter because they show the design
**scales** to millions of claims. In the report, measure and state the average search time per
claim.

---

## 17. Algorithms and Models Explained

This section explains every important algorithm and model in the project: **what** it does,
**how** it works, **where** we use it, and **where to learn more**.

### 17.0 Map: which algorithm is used at which step

```
 STEP                         ALGORITHM / MODEL USED                                   §
 ─────────────────────────────────────────────────────────────────────────────────────────
 Preprocessing                Regex text cleaning                                      A1
                              Leakage fix (count subtraction)                          A2
                              Named Entity Recognition (spaCy)                         A3
                              Sentence segmentation (spaCy)                            A4
 Embeddings                   WordPiece tokenization                                   A5
                              Transformer + self-attention                             A6
                              Sentence embedding model: all-MiniLM-L6-v2               A7
 Building the graph           MERGE (match-or-create) + uniqueness constraints         A8
                              HNSW vector index                                        A9
 Retrieval                    Cosine similarity                                        A10
                              Jaccard similarity                                       A11
                              Weighted hybrid score                                    A12
                              Label-balanced top-k selection                           A13
                              Grid search for the weights                              A14
 Prediction                   LLM in-context learning (Llama 3.1 8B), greedy decoding  A15
 Evaluation                   Accuracy, Precision, Recall, F1, MCC                     A16
 Live demo (optional)         k-nearest-neighbour subject voting                       A17
```

---

### A1. Text Cleaning with Regular Expressions

**What:** Removes noise so the same text always looks the same.

**How:** A *regular expression* (regex) is a search pattern. Examples we use:

| Pattern | Meaning | Effect |
|---|---|---|
| `\s+` → `" "` | one or more whitespace characters | `"Last  year\r\nwas"` → `"Last year was"` |
| `^(Says\|Said)( that)?\s+` | "Says" / "Said" / "Says that" at the very start | `"Says crime rose"` → `"crime rose"` |

We also lowercase names (`"Thom Tillis"` → `"thom tillis"`) so one person is always one node.

**Why it matters:** without cleaning, `"Thom Tillis"` and `"thom tillis "` would become **two
different Speaker nodes**, breaking the connections in the graph.

**Learn more:** <https://regexone.com> (interactive) · <https://docs.python.org/3/howto/regex.html>

---

### A2. Leakage Fix (Count Subtraction)

**What:** Removes hidden "answer" information from the speaker history columns.

**How:** Each row has six counts (`pants_on_fire_counts` … `true_counts`). If they include the
current claim's own verdict, a model could read the answer from them. For each row:

```
own_column = count_columns[label]        # e.g. label 1 (False) → false_counts
row[own_column] = max(0, row[own_column] - 1)
```

**Example:** a claim with label 1 (False) and `false_counts = 5` → becomes 4. The current claim
no longer "votes" for its own answer.

**Learn more (what data leakage is):** <https://www.kaggle.com/code/alexisbcook/data-leakage>

---

### A3. Named Entity Recognition (NER) — spaCy

**What:** Finds names of people, places, organisations, laws etc. inside a sentence.

```
"Scott Walker's tax cuts cost Wisconsin 50,000 jobs."
  └── Scott Walker → PERSON      └── Wisconsin → GPE (country/state/city)    50,000 → CARDINAL
```

**Model:** spaCy `en_core_web_sm` (small English pipeline). It is trained on the **OntoNotes 5**
corpus and recognises **18 entity types**. We keep only: `PERSON`, `ORG`, `GPE`, `NORP`
(nationalities/religious/political groups), `LOC`, `LAW`, `EVENT`. Numbers and dates are dropped
because "50,000" in two claims does not make them related.

**How it works (simplified):**

1. **Token vectors:** each word is turned into a vector using hashed word features (the word,
   its prefix, suffix and shape like `Xxxxx`) passed through a small **convolutional neural
   network (CNN)** that also looks at the neighbouring words.
2. **Transition-based tagging:** the model reads the sentence left to right and, at each word,
   chooses an action such as *BEGIN an entity*, *INSIDE the entity*, *LAST word of the entity*,
   *OUTSIDE any entity*, together with the entity type. The choice is made by a neural network
   using the token vectors.

   ```
   Scott   → BEGIN-PERSON
   Walker  → LAST-PERSON
   's      → OUT
   tax     → OUT
   ...
   Wisconsin → UNIT-GPE   (single-word entity)
   ```

**Limitations:** the small model makes mistakes (misses some entities, or confuses types). A
larger model (`en_core_web_trf`, transformer-based) is more accurate but much slower — a
possible improvement.

**Learn more:**
- spaCy free course: <https://course.spacy.io>
- NER docs: <https://spacy.io/usage/linguistic-features#named-entities>
- Model details: <https://spacy.io/models/en>
- How the parser/NER works: <https://explosion.ai/blog/parsing-english-in-python>

---

### A4. Sentence Segmentation — spaCy

**What:** Splits the journalist's justification into separate sentences → one **Evidence** node
each.

**How:** In `en_core_web_sm`, sentence boundaries come from the **dependency parser**: it builds a
grammar tree for the text, and each tree root starts a new sentence. This is smarter than
splitting on `"."` — it does not break `"U.S. Senator"` or `"3.5 percent"` into two sentences.

We drop very short pieces (< 20 characters), which are usually fragments.

**Learn more:** <https://spacy.io/usage/linguistic-features#sbd>

---

### A5. Tokenization — WordPiece

**What:** Before a transformer can read text, the text is cut into **tokens** (pieces of words)
and each token is replaced by a number (its ID in a vocabulary).

**How:** all-MiniLM-L6-v2 uses BERT's **WordPiece** tokenizer: lowercase vocabulary of
**30,522** pieces. Common words are one token; rare words are split into known pieces (`##`
means "continues the previous piece"):

```
"background checks"  → ["background", "checks"]
"Obamacare"          → ["obama", "##care"]        (example of a split)
```

Special tokens are added: `[CLS]` at the start and `[SEP]` at the end.

**Why it matters:** the model can handle **any word**, even ones it never saw, by combining
pieces. Max input length is **256 tokens**; LIAR2 claims are ~20–30 words, far below that limit.

**Learn more:** Hugging Face course, tokenizers chapter:
<https://huggingface.co/learn/llm-course/chapter6/6>

---

### A6. Transformer and Self-Attention

**What:** The neural network architecture behind both our embedding model and the LLM.

**Key idea — self-attention:** every word looks at every other word in the sentence and decides
how much each one matters for understanding it.

```
"I sat by the bank of the river"
                ▲            │
                └────────────┘  "bank" pays strong attention to "river"
                                 → it understands "bank" = riverside, not money
```

**How (the formula):** each token vector produces three vectors — a **Query** (what am I looking
for?), a **Key** (what do I contain?) and a **Value** (what information do I pass on?).

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d ) · V
```

1. `Q · Kᵀ` — score how well each word's query matches each other word's key.
2. `/ √d` — scale down so numbers stay stable (`d` = vector size).
3. `softmax` — turn scores into percentages that add up to 1.
4. `· V` — each word's new vector = weighted mix of all words' values.

**Multi-head attention** runs several attentions in parallel (each "head" can focus on a
different relationship, e.g. grammar vs meaning). A **transformer layer** = multi-head attention
+ a small feed-forward network. Stacking layers gives deeper understanding.

| Model | Type | Layers | Vector size | Used for |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | **Encoder** (reads whole sentence at once) | 6 | 384 | Embeddings |
| Llama 3.1 8B | **Decoder** (writes text one token at a time) | 32 | 4,096 | Final FAKE/REAL answer |

**Learn more:**
- The Illustrated Transformer (best visual intro): <https://jalammar.github.io/illustrated-transformer/>
- 3Blue1Brown video series on transformers: <https://www.3blue1brown.com/topics/neural-networks>
- Original paper, *Attention Is All You Need*: <https://arxiv.org/abs/1706.03762>
- The Illustrated BERT: <https://jalammar.github.io/illustrated-bert/>

---

### A7. The Embedding Model: `all-MiniLM-L6-v2` ⭐

This is the model that turns every claim into a list of numbers. Same model as the reference
paper.

#### Facts at a glance

| Property | Value |
|---|---|
| Name | `sentence-transformers/all-MiniLM-L6-v2` |
| Library | `sentence-transformers` |
| Output | **384 numbers** per sentence (a 384-dimensional vector) |
| Architecture | BERT-style **encoder**, **6** transformer layers, hidden size **384** |
| Size | ~22 million parameters (~90 MB) — runs fine on a laptop CPU |
| Origin | `nreimers/MiniLM-L6-H384-uncased`: Microsoft's 12-layer MiniLM with **every second layer kept** |
| Tokenizer | WordPiece, lowercase ([A5](#a5-tokenization--wordpiece)) |
| Max input | **256** tokens (longer text is cut off); trained with 128 |
| Pooling | **Mean pooling** |
| Normalisation | **L2** (every vector has length 1) |
| Training data | **~1.17 billion** sentence pairs from 28 datasets (Reddit, Q&A sites, paper titles, NLI, etc.) |
| Training objective | **Contrastive** (in-batch negatives, cross-entropy) |
| Model page | <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2> |

#### How a sentence becomes 384 numbers

```
"Gun deaths fell in 2016"
        │  A5 WordPiece
        ▼
[CLS] gun deaths fell in 2016 [SEP]          ← 7 tokens
        │  6 transformer layers (A6)
        ▼
7 vectors, each with 384 numbers             ← one per token, each "aware" of the others
        │  Mean pooling
        ▼
1 vector of 384 numbers                      ← the average of the token vectors
        │  L2 normalisation
        ▼
[0.021, -0.113, 0.045, …, 0.067]  (length = 1)   ← the claim's embedding
```

**Mean pooling (toy example with 3 numbers instead of 384):**

```
token "guns" → (1, 0, 2)
token "kill" → (3, 2, 0)
mean         → ((1+3)/2, (0+2)/2, (2+0)/2) = (2, 1, 1)
```

Padding tokens (added to make sentences in a batch equally long) are ignored using the
*attention mask*.

**L2 normalisation:** divide the vector by its length so its length becomes 1.

```
(2, 1, 1) has length √(2² + 1² + 1²) = √6 ≈ 2.449
normalised = (2/2.449, 1/2.449, 1/2.449) = (0.816, 0.408, 0.408)   → length 1
```

After normalisation, **cosine similarity = a simple dot product** ([A10](#a10-cosine-similarity)),
which is fast.

#### How the model learned "meaning" — contrastive training

The model was trained on pairs of texts that mean the same thing or belong together (a question
and its answer, a title and its abstract, two paraphrases):

```
Batch of 1,024 pairs:
  (Q1 "How do I cook rice?",        A1 "Boil water, add rice…")
  (Q2 "Who won the 2010 World Cup?", A2 "Spain won in 2010…")
  ...
For Q1: A1 is the RIGHT partner, A2…A1024 are WRONG partners ("in-batch negatives").
Training goal: make cosine(Q1, A1) high and cosine(Q1, every other A) low.
```

Technically: compute the cosine of Q1 with all 1,024 answers, treat it as a 1,024-way
classification ("which one is my partner?") and apply **cross-entropy loss**. After ~100k
training steps on 1 billion+ pairs, sentences with similar meaning land close together in the
384-dimensional space.

#### Why 384 dimensions?

- 384 is simply the model's hidden size. Each number captures some (not human-readable) aspect
  of meaning.
- **More dimensions** (e.g. 768 or 1,024) can capture more detail but need more memory and are
  slower.
- **Storage for our project:** 18,400 claims × 384 numbers × 4 bytes ≈ **28 MB** — tiny.

#### Why this model?

1. **Same as the reference paper** → fair comparison.
2. **Small and fast** → embeds all 18,400 claims in a few minutes on a normal laptop.
3. Claims are short sentences, exactly what it was trained for.

**Possible upgrade (ablation experiment):** `BAAI/bge-base-en-v1.5` (768 dimensions, usually more
accurate). Changing the model only requires changing the model name and the index dimension
(384 → 768). Compare retrieval quality and final F1.

#### Code

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(["Gun deaths fell in 2016"], normalize_embeddings=True)
print(emb.shape)   # (1, 384)
```

**Learn more:**
- Sentence-BERT paper (how sentence embeddings are trained): <https://arxiv.org/abs/1908.10084>
- SBERT documentation and tutorials: <https://www.sbert.net>
- The loss used (MultipleNegativesRankingLoss): <https://sbert.net/docs/package_reference/sentence_transformer/losses.html>
- MiniLM paper (how the small model was distilled from a big one): <https://arxiv.org/abs/2002.10957>
- Compare embedding models: MTEB leaderboard <https://huggingface.co/spaces/mteb/leaderboard>

---

### A8. Graph Construction: MERGE (Match-or-Create)

**What:** Builds one connected graph without duplicate nodes.

**How:** `MERGE (s:Speaker {name: "chris abele"})`:

```
Does a Speaker node with name "chris abele" exist?
   ├── YES → reuse it
   └── NO  → create it
```

Running the same loading script twice does **not** create duplicates (the operation is
*idempotent*). This is exactly what links 18,400 separate rows into one network.

**Uniqueness constraints** ([Section 16](#16-making-search-fast), Trick 3) create an index so
the "does it exist?" check is instant instead of scanning every node.

**Learn more:**
- MERGE docs: <https://neo4j.com/docs/cypher-manual/current/clauses/merge/>
- Free Neo4j GraphAcademy courses (Cypher fundamentals, graph data modelling):
  <https://graphacademy.neo4j.com>

---

### A9. HNSW — The Vector Index ⭐

**What:** Finds the 50 most similar claims to a query **without comparing it to every claim**.
Neo4j's vector index uses **HNSW (Hierarchical Navigable Small World)** graphs, implemented by
Apache Lucene.

**Idea — a multi-level map:** think of travelling to a house in another city. You don't walk
every street; you take a flight (big jumps), then a train, then a bus, then walk (small steps).
HNSW builds such a "transport network" over all embeddings.

```
Layer 2:  ●───────────────────●                  few nodes, long links (flights)
          │                   │
Layer 1:  ●──────●─────●──────●────●             more nodes (trains)
          │      │     │      │    │
Layer 0:  ●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●        ALL nodes, short links (walking)
```

**Building the index (done once):**

1. Each new vector gets a random top layer — most stay only in layer 0, a few reach layer 1, very
   few reach layer 2 (probability drops by a constant factor per level).
2. In every layer it belongs to, the vector is linked to its **M** nearest neighbours.
   Neo4j default: **`vector.hnsw.m = 16`**.
3. While searching for those neighbours during insertion, the algorithm keeps a candidate list
   of size **`vector.hnsw.ef_construction`** (Neo4j default **100**). Bigger = better index,
   slower build.

**Searching (done for every query):**

1. Start at the entry point in the **top layer**.
2. **Greedy step:** move to whichever neighbour is closer to the query; repeat until no neighbour
   is closer.
3. Drop down one layer and continue from there.
4. In layer 0, do a wider search keeping a list of the best candidates, and return the top **k**
   (we use k = 50).

**Speed:** the number of steps grows roughly with the **logarithm** of the number of vectors —
for 18,400 claims there are only about 3–4 layers. 10× more data adds only a few more steps.

**Trade-off:** HNSW is **approximate** — it may occasionally miss one of the true 50 nearest
neighbours. For our use this is fine (we re-score the 50 candidates anyway).

**Neo4j specifics you must know:**

- Neo4j returns a **rescaled** cosine score: `score = (1 + cosine) / 2`, always between 0 and 1.
  We convert back with `cosine = 2 × score − 1` (already done in the
  [Section 10](#10-retrieval-the-scoring-formula-with-example) query).
- Supported dimensions: 1 – 4,096.
- Query (Neo4j 5): `CALL db.index.vector.queryNodes('claim_emb', 50, $query_emb)`.
  Newer Neo4j versions (2026.x) prefer the `SEARCH` clause:

  ```cypher
  MATCH (c:Claim)
    SEARCH c IN (VECTOR INDEX claim_emb FOR $query_emb LIMIT 50) SCORE AS score
  RETURN c.id, score
  ```

**Learn more:**
- Original HNSW paper (Malkov & Yashunin): <https://arxiv.org/abs/1603.09320>
- Visual explanation (Pinecone): <https://www.pinecone.io/learn/series/faiss/hnsw/>
- Neo4j vector index docs: <https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/>

---

### A10. Cosine Similarity

**What:** Measures how similar two embeddings are by the **angle** between them.

```
                    a · b            sum of (aᵢ × bᵢ)
cosine(a, b) = ─────────────── = ─────────────────────────
                 |a| × |b|        length(a) × length(b)
```

| Value | Meaning |
|---|---|
| 1 | Same direction → same meaning |
| 0 | Unrelated |
| −1 | Opposite direction (rare for sentence embeddings) |

**Toy example (3 numbers instead of 384):**

```
a = (1, 2, 2)   b = (2, 1, 2)
a · b = 1×2 + 2×1 + 2×2 = 8
|a| = √(1+4+4) = 3     |b| = √(4+1+4) = 3
cosine = 8 / (3 × 3) = 0.889   → very similar
```

**Shortcut:** our embeddings are already normalised to length 1 ([A7](#a7-the-embedding-model-all-minilm-l6-v2-)),
so `cosine = a · b` — just multiply and add.

**Why cosine and not plain distance?** It compares *direction* (meaning), not *size*, so it is
not affected by how long or short the sentence is.

**Learn more:** <https://www.pinecone.io/learn/vector-similarity/> ·
<https://en.wikipedia.org/wiki/Cosine_similarity>

---

### A11. Jaccard Similarity (Shared Entities)

**What:** Measures overlap between two **sets** (here: the entity sets of two claims).

```
                 |A ∩ B|          items in BOTH sets
Jaccard(A, B) = ───────── = ─────────────────────────────
                 |A ∪ B|     all DIFFERENT items together
```

**Examples:**

```
A = {scott walker, wisconsin}   B = {wisconsin, scott walker}  → 2 / 2 = 1.0
A = {scott walker, wisconsin}   B = {wisconsin, texas}         → 1 / 3 = 0.33
A = {scott walker, wisconsin}   B = {texas}                    → 0 / 3 = 0.0
A = {}                          B = {}                         → defined as 0 (nothing to compare)
```

**Why divide by the union?** Without it, a claim mentioning 20 entities would match almost
everything. Dividing by the union rewards claims that are **specifically** about the same things.

**Learn more:** <https://en.wikipedia.org/wiki/Jaccard_index>

---

### A12. Weighted Hybrid Score

**What:** Combines the three signals into one number per candidate
(full worked example in [Section 10](#10-retrieval-the-scoring-formula-with-example)):

```
score = w₁ × cosine  +  w₂ × same_speaker_or_subject  +  w₃ × jaccard_entities
        (0.6)            (0.2)                           (0.2)          w₁ + w₂ + w₃ = 1
```

**How it is different from the reference paper:**

| | Paper | Ours |
|---|---|---|
| Formula | `α·cosine + β·S(e)` | `w₁·cosine + w₂·link + w₃·Jaccard` |
| Structural part depends on the new claim? | ❌ No — `S(e)` = 1 for almost every stored event | ✅ Yes — compares the candidate **with the new claim** |
| Effect | Ranking ≈ plain cosine | Graph connections actually change the ranking |

All three parts are between 0 and 1 (after converting the Neo4j score, [A9](#a9-hnsw--the-vector-index-)),
so the weights directly express how much each part matters.

**Learn more:** this is a form of *score fusion* used in hybrid search —
<https://www.pinecone.io/learn/hybrid-search-intro/>

---

### A13. Label-Balanced Top-k Selection

**What:** Picks the examples shown to the LLM so both labels are equally represented
(reasoning in [Section 14, Q1](#14-design-decisions-faq)).

**Algorithm:**

```
input:  candidates (each with score and label), k = 10
1. fake = candidates with label FAKE, sorted by score (high → low)
2. real = candidates with label REAL, sorted by score (high → low)
3. chosen = first k/2 of fake  +  first k/2 of real
4. if one list has fewer than k/2 items:
       ask the vector index for more candidates (e.g. 100 instead of 50) and repeat;
       if still not enough, fill the gap from the other list
5. shuffle chosen        ← so the LLM does not learn "FAKE examples always come first"
output: chosen
```

**Learn more (why example order and balance affect LLMs):** *Calibrate Before Use* —
<https://arxiv.org/abs/2102.09690>

---

### A14. Grid Search for the Weights

**What:** Systematically tries weight combinations and keeps the best one, measured on the
**validation** set.

**Algorithm:**

```
for w1 in 0.4, 0.5, …, 1.0:
    for w2 in 0.0, 0.1, …, (1 − w1):
        w3 = 1 − w1 − w2
        run retrieval with (w1, w2, w3) on the validation claims
        measure quality
keep the (w1, w2, w3) with the best quality → use it on the test set
```

With steps of 0.1 and `w1 ≥ 0.4` there are **28** combinations.

**Practical problem:** running the LLM on 2,300 validation claims × 28 combinations is far too
many LLM calls. **Two-stage solution:**

1. **Cheap proxy (no LLM):** for each validation claim, take the top 10 retrieved claims
   (*not* balanced) and let them **vote** — the majority label is the prediction. Measure F1 of
   this vote for every weight combination. Takes seconds. Better retrieval → better vote.
2. **Confirm with the LLM:** run the full system only for the best 2–3 combinations on a fixed
   sample of ~300 validation claims, and pick the winner.

**Learn more:** <https://scikit-learn.org/stable/modules/grid_search.html>

---

### A15. The LLM: Llama 3.1 8B Instruct + In-Context Learning ⭐

#### Facts at a glance

| Property | Value |
|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Parameters | 8 billion |
| Architecture | Decoder-only transformer, 32 layers, hidden size 4,096, **Grouped-Query Attention (GQA)** |
| Context window | 128k tokens (the provider may set a lower limit) |
| Training | Pre-trained on ~15 trillion tokens; instruction-tuned with **SFT** and **RLHF** |
| Knowledge cutoff | December 2023 |
| License | Llama 3.1 Community License |
| Model page | <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> |

#### How it generates an answer

A decoder LLM **predicts the next token** again and again:

```
prompt … "Answer in JSON:"  → predicts  {
                            → predicts  "label"
                            → predicts  :
                            → predicts  "FAKE"   … and so on until the JSON is complete
```

At each step it computes a probability for every possible next token.

- **Temperature = 0 (greedy decoding):** always pick the most likely token → the same prompt
  always gives the same answer. Needed for repeatable experiments.
- **GQA:** several attention heads share the same Keys/Values, which saves memory and makes
  generation faster with little loss in quality.

#### In-context (few-shot) learning — the core of RAG

We **never retrain** the LLM. Instead, the 10 retrieved examples and their evidence are placed
**inside the prompt**, and the model learns the task pattern from them on the fly. This is why
retrieval quality matters so much: better examples in the prompt → better answers.

**Prompt size estimate:** 10 examples × (claim ~30 tokens + ~3 evidence sentences ~75 tokens) +
instructions ≈ **1,200–1,500 tokens** per claim — well within the context window.

#### Output handling

```
1. Parse the response as JSON.
2. If parsing fails → retry once with "Return ONLY valid JSON."
3. If it fails again → record the claim as "unparsed" and count it as wrong in the evaluation
   (report how many there were).
```

#### ⚠️ Caveat: memorisation (data contamination)

The model's knowledge goes up to December 2023, and PolitiFact articles are on the public web.
The LLM may have **already seen some verdicts** during training, which can make the
*LLM alone* baseline look stronger than it really is. Report this honestly. A good extra
analysis: compare accuracy on claims dated **before** vs **after** December 2023 (if LIAR2 has
enough later claims).

**Learn more:**
- *Language Models are Few-Shot Learners* (GPT-3, in-context learning): <https://arxiv.org/abs/2005.14165>
- *The Llama 3 Herd of Models*: <https://arxiv.org/abs/2407.21783>
- Original RAG paper: <https://arxiv.org/abs/2005.11401>
- Microsoft GraphRAG paper: <https://arxiv.org/abs/2404.16130>
- Hugging Face LLM course (free): <https://huggingface.co/learn/llm-course>
- Prompting guide: <https://www.promptingguide.ai>
- GQA paper: <https://arxiv.org/abs/2305.13245>

---

### A16. Evaluation Metrics

Treat **FAKE as the positive class** and count four outcomes:

|  | Truly FAKE | Truly REAL |
|---|---|---|
| **Predicted FAKE** | TP (true positive) | FP (false positive) |
| **Predicted REAL** | FN (false negative) | TN (true negative) |

| Metric | Formula | Question it answers |
|---|---|---|
| Accuracy | (TP + TN) / all | How many did we get right overall? |
| Precision | TP / (TP + FP) | When we say FAKE, how often are we right? |
| Recall | TP / (TP + FN) | Of all FAKE claims, how many did we catch? |
| F1 | 2 × P × R / (P + R) | Balance of precision and recall |
| MCC | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Overall quality from −1 to +1; fair even when classes are unbalanced |

**Worked example (100 test claims):** TP = 40, FP = 15, FN = 10, TN = 35

```
Accuracy  = (40 + 35) / 100            = 0.75
Precision = 40 / (40 + 15)             = 0.727
Recall    = 40 / (40 + 10)             = 0.800
F1        = 2 × 0.727 × 0.8 / 1.527    = 0.762
MCC       = (40×35 − 15×10) / √(55 × 50 × 50 × 45)
          = 1250 / 2487.5              = 0.503
```

**Why MCC?** If 70% of claims were FAKE, a model that *always* says FAKE gets 70% accuracy and a
high F1 — but MCC = 0, exposing that it learned nothing. The reference paper also reports MCC.

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
```

**Learn more:**
- scikit-learn metrics guide: <https://scikit-learn.org/stable/modules/model_evaluation.html>
- Google ML crash course (precision/recall): <https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall>
- Why MCC is better than F1 and accuracy (Chicco & Jurman, 2020): <https://doi.org/10.1186/s12864-019-6413-7>

---

### A17. k-Nearest-Neighbour Subject Voting (live demo only)

**What:** Guesses the subjects of a brand-new claim that has no `subject` column
([Section 14, Q3](#14-design-decisions-faq), Option A).

**Algorithm:**

```
1. Embed the new claim (A7) and find its 10 nearest training claims (A9).
2. Count how often each subject appears among those 10.
3. Keep subjects that appear in at least 3 of the 10 (maximum 3 subjects).
```

**Example:** neighbours' subjects → guns ×7, crime ×4, polls ×2 → predicted subjects:
**guns, crime**.

**Learn more:** <https://scikit-learn.org/stable/modules/neighbors.html>

---

### Recommended Learning Path for Beginners

| Order | Topic | Resource |
|---|---|---|
| 1 | Python + pandas basics | <https://www.kaggle.com/learn/pandas> |
| 2 | spaCy (NER, sentences) | <https://course.spacy.io> |
| 3 | Transformers (visual) | <https://jalammar.github.io/illustrated-transformer/> |
| 4 | Embeddings and similarity | <https://www.sbert.net> (Quickstart + Semantic Search) |
| 5 | Neo4j + Cypher | <https://graphacademy.neo4j.com> (Neo4j Fundamentals, Cypher Fundamentals) |
| 6 | Vector search / HNSW | <https://www.pinecone.io/learn/series/faiss/hnsw/> |
| 7 | LLMs and prompting | <https://huggingface.co/learn/llm-course> · <https://www.promptingguide.ai> |
| 8 | RAG + GraphRAG with Neo4j | <https://graphacademy.neo4j.com> (LLM / GraphRAG courses) |
| 9 | Evaluation metrics | <https://developers.google.com/machine-learning/crash-course/classification> |

---

## 18. Project Folder Structure

```
IRE/
├── README.md                 ← this file
├── ire_paper.pdf             ← reference paper
├── requirements.txt
├── data/
│   ├── raw/                  ← downloaded LIAR2 (optional local copy)
│   └── processed/            ← cleaned data with entities + embeddings
├── notebooks/
│   └── 01_explore_data.ipynb ← charts: label counts, top speakers, top subjects
├── src/
│   ├── preprocess.py         ← Section 8
│   ├── build_graph.py        ← Section 9
│   ├── retrieve.py           ← Section 10
│   ├── llm.py                ← Section 11
│   └── evaluate.py           ← Section 12
└── results/
    ├── predictions_*.csv
    └── metrics.csv
```

---

## 19. Setup and Installation

1. Install **Python 3.10+** and **Docker Desktop**.
2. Create a virtual environment and install packages:

   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   pip install datasets pandas scikit-learn spacy sentence-transformers neo4j groq
   python -m spacy download en_core_web_sm
   ```

3. Start Neo4j (see [Section 9](#9-building-the-graph-in-neo4j)).
4. Get a free Groq API key from <https://console.groq.com> and set it:

   ```bash
   set GROQ_API_KEY=your_key_here    # Windows cmd
   ```

5. Run the pipeline:

   ```bash
   python src/preprocess.py
   python src/build_graph.py
   python src/evaluate.py
   ```

---

## 20. Timeline

| Week | Task | Output |
|---|---|---|
| 1 | Setup; load LIAR2; explore the data | Notebook with charts |
| 2 | Preprocessing (Section 8) | `data/processed/` files |
| 3 | Build the graph in Neo4j | Graph visible in browser |
| 4 | Retrieval + LLM prompt | Predictions for a few claims |
| 5 | Run all 3 systems; compute metrics | Results table |
| 6 | Optional cause/effect extra; report and presentation | Final report |

---

## 21. What We Are NOT Doing (Keeping It Simple)

- Using multiple datasets — **LIAR2 only**.
- 6-class prediction — we start with **FAKE / REAL**.
- Training our own neural networks — we only **use** ready-made models
  (spaCy, MiniLM, Llama).
- Advanced entity linking (e.g. Wikidata) or clustering — possible future work.

---

## 22. References and Links

- **Reference paper:** Haque et al., *GraphRAG-Causal: A Novel Graph-Augmented Framework for
  Causal Reasoning and Annotation in News*, arXiv:2506.11600 (2025) — [ire_paper.pdf](ire_paper.pdf)
- **LIAR2 dataset:** <https://huggingface.co/datasets/chengxuphd/liar2> ·
  <https://github.com/chengxuphd/liar2>
- **Original LIAR dataset:** <https://huggingface.co/datasets/ucsbnlp/liar>
- **Causal News Corpus** (used by the paper): <https://github.com/tanfiona/CausalNewsCorpus>
- **Neo4j:** <https://neo4j.com> · Cypher manual: <https://neo4j.com/docs/cypher-manual/current/>
- **Sentence-Transformers (all-MiniLM-L6-v2):**
  <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- **spaCy:** <https://spacy.io>
- **Groq API:** <https://groq.com> · **Ollama:** <https://ollama.com>

### Possible future datasets (not needed now)

- PUBHEALTH (health claims): <https://huggingface.co/datasets/ImperialCollegeLondon/health_fact>
- AVeriTeC (real-world claims with evidence): <https://huggingface.co/chenxwh/AVeriTeC>
- FakeNewsNet: <https://github.com/KaiDMML/FakeNewsNet>
