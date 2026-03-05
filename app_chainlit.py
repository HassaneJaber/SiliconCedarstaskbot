import uuid
import chainlit as cl

from project1.graphs.supervisor_graph import run_supervisor

# Optional (recommended): real chart rendering
try:
    import plotly.graph_objects as go
except Exception:
    go = None


def _plotly_from_chartjs(cfg: dict):
    """
    Convert your Chart.js-like config (type/labels/data) into a Plotly Figure.
    """
    if go is None or not cfg:
        return None

    chart_type = (cfg.get("type") or "bar").lower()
    labels = (cfg.get("data", {}).get("labels") or [])
    datasets = (cfg.get("data", {}).get("datasets") or [])
    values = (datasets[0].get("data") if datasets else []) or []
    ds_label = (datasets[0].get("label") if datasets else "Value") or "Value"

    if chart_type == "line":
        fig = go.Figure([go.Scatter(x=labels, y=values, mode="lines+markers", name=ds_label)])
    elif chart_type in ["pie", "doughnut"]:
        hole = 0.4 if chart_type == "doughnut" else 0.0
        fig = go.Figure([go.Pie(labels=labels, values=values, hole=hole, name=ds_label)])
    else:  # default bar
        fig = go.Figure([go.Bar(x=labels, y=values, name=ds_label)])

    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig


@cl.on_chat_start
async def on_chat_start():
    thread_id = str(uuid.uuid4())[:8]
    cl.user_session.set("thread_id", thread_id)
    await cl.Message(content=f"✅ Project 1 Supervisor ready. (thread_id={thread_id})").send()


@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id") or "chainlit"
    out = run_supervisor(message.content, thread_id=thread_id)

    route = out.get("route", "unknown")
    response = out.get("response", "")
    viz_cfg = out.get("viz_config")

    elements = []
    if viz_cfg:
        fig = _plotly_from_chartjs(viz_cfg)
        if fig is not None:
            elements.append(cl.Plotly(name="chart", figure=fig, display="inline"))

    content = f"*route:* `{route}`\n\n{response}"

    if viz_cfg and go is None:
        content += "\n\n_(Chart generated, but Plotly is not installed to render it in the UI.)_"

    await cl.Message(
        content=content,
        elements=elements if elements else None,
    ).send()