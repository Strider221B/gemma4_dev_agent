---
name: repo_navigation
description: Guidance on efficient code search and graph-based repository navigation.
---

# Repository Navigation Skill

This guide explains how to effectively explore unfamiliar Python codebases using code intelligence tools and graph operations.

## 1. Tool Selection Matrix

| Objective | Recommended Tool | When to Use |
|---|---|---|
| Initial discovery | `search_similar_code` | You have a concept, keyword, error message, or symbol name and need entry points. |
| Call & hierarchy traversal | `get_code_neighbors` | You identified a symbol (function, method, class) and need direct callers, callees, or inheritance links. |
| Subsystem mapping | `get_code_subgraph` | You need a multi-hop subgraph around a core symbol to understand module architecture. |
| Content inspection | `read_file` | You know the exact file and line range; only read necessary lines (never entire large files). |

## 2. Navigation Strategies

### Step 1: Entry Point Identification
- Search for the specific function, class, or exception referenced in the problem description using `search_similar_code`.
- Query with descriptive keywords or exact symbol names.
- Focus on top-scoring results in production modules rather than test suites.

### Step 2: Neighbor Exploration
- Once a target symbol is identified, call `get_code_neighbors` on its node ID.
- Examine inbound edges (who calls or instantiates this) to understand execution flow.
- Examine outbound edges (what helper methods or external dependencies are invoked) to locate the point of failure.

### Step 3: Targeted Reading
- Use line offsets returned by the graph/search tools to call `read_file(filepath, start_line, end_line)`.
- Restrict reading to 30-50 lines per call around critical logic.
- Avoid broad grep searches across the workspace when graph tools are available.

## 3. Interpreting Graph Results
- **Node ID format**: Typically `module.path:Class.method` or `module.path:function`.
- **Edge types**:
  - `calls`: Direct function or method invocation.
  - `defines`: Scope hierarchy (module defines class, class defines method).
  - `inherits`: Class inheritance relationship.
  - `references`: Attribute access or symbol reference.

## 4. Delegation to Code Analyzer Sub-Agent
- Delegate exploration to `code_analyzer` when:
  - The problem statement involves multiple interconnected modules.
  - The root cause location is unknown and requires multi-step investigation.
  - You need to preserve root coder context window space while researching call hierarchies.
- Do NOT delegate when:
  - The exact file and line number are already stated in the issue description.
  - You are applying an edit or running verification tests.
