"""
Flask backend for the Multi-Agent Research System.
Same 4 steps as pipeline.py, but each step is streamed to the browser
(Server-Sent Events) so the UI can show live progress.
"""
import json
import os
import traceback

from flask import Flask, Response, jsonify, request, send_from_directory

from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain

app = Flask(__name__, static_folder=".", static_url_path="")


def to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            parts.append(c.get("text", "") if isinstance(c, dict) else str(c))
        return "".join(parts)
    return str(content)


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/api/research")
def research():
    topic = (request.args.get("topic") or "").strip()
    if len(topic) < 3:
        return jsonify(error="Please enter a topic with at least 3 characters."), 400

    def generate():
        try:
            yield sse({"type": "step", "step": 1, "status": "running"})
            search_agent = build_search_agent()
            res = search_agent.invoke({
                "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
            })
            search_results = to_text(res["messages"][-1].content)
            yield sse({"type": "step", "step": 1, "status": "done", "data": search_results})

            yield sse({"type": "step", "step": 2, "status": "running"})
            reader_agent = build_reader_agent()
            res = reader_agent.invoke({
                "messages": [("user",
                    f"Based on the following search results about '{topic}', "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{search_results[:800]}"
                )]
            })
            scraped = to_text(res["messages"][-1].content)
            yield sse({"type": "step", "step": 2, "status": "done", "data": scraped})

            yield sse({"type": "step", "step": 3, "status": "running"})
            research_combined = (
                f"SEARCH RESULTS : \n {search_results} \n\n"
                f"DETAILED SCRAPED CONTENT : \n {scraped}"
            )
            report = writer_chain.invoke({"topic": topic, "research": research_combined})
            yield sse({"type": "step", "step": 3, "status": "done", "data": report})

            yield sse({"type": "step", "step": 4, "status": "running"})
            feedback = critic_chain.invoke({"report": report})
            yield sse({"type": "step", "step": 4, "status": "done", "data": feedback})

            yield sse({"type": "done"})
        except Exception as e:
            traceback.print_exc()
            yield sse({"type": "error", "message": str(e)})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)