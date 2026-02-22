# Smoke test checklist (manual)

Run Chainlit:

```bash
python -m chainlit run app_chainlit.py -w
```

Use these prompts in the UI (same thread):

## Conversation route
- `Explain what your supervisor does in one paragraph.`

## RAG route (docs/)
- `Based on docs, summarize what the sample document says.`

## Viz route (data provided)
- `Make a pie chart for A=30, B=70.`
- `Chart.js line chart points: 1,3,2,5.`

## SQL route (database)
- `What tables do I have in my database?`
- `Run SQL: SELECT * FROM demo_users`
- `Show me the average age in demo_users`

## SQL -> Viz auto-handoff
- `Make a bar chart of ages from demo_users`

## Research route
- `Research: what are common RAG evaluation metrics?` (should return a plan + report, no fake URLs)
