# Weights

Download the PromptPilot weight bundle from the Releases page and extract it
into the project root. This directory should contain:

```text
weights/
|-- best-strict.pt
|-- best-graph.pt
`-- best-policy-table.pkl
```

`best-strict.pt` is used by `--mode pt-strict`, `best-graph.pt` by
`--mode pt-graph`, and `best-policy-table.pkl` by `--mode pkl`.
