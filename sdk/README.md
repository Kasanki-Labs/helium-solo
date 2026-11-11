# Kasanki Helium Python SDK

Helium records, replays, tests and fixes safe AI agents. This SDK records AI agent runs for the Kasanki Helium engine, so every run can be checked, replayed and tested.

```python
import kasanki_helium as helium
from openai import OpenAI

client = OpenAI()

with helium.run("support-agent", input=ticket) as run:
    llm = run.wrap(client)
    now = run.now()
    hits = run.call_tool("search", search, query="refund policy")
    reply = llm.chat.completions.create(model="gpt-5", messages=messages)
    run.set_output(answer)
```

`run.wrap(client)` records each model call into that run, streamed or not, sync and async:

- **OpenAI** (`openai` 3.x): chat completions, responses and embeddings (also their `.stream` helpers), from `OpenAI`, `AsyncOpenAI` and the clients built on them (Azure, Bedrock, and OpenAI-compatible servers such as vLLM or Ollama).
- **Anthropic** (`anthropic` 2.x): `messages.create`, `.parse`, `.stream` and `.count_tokens`, also under `beta.messages`, from any of the package's clients (`Anthropic`, `AsyncAnthropic`, Bedrock, Vertex, Foundry…).
- **Google** (`google-genai` 2.18+): `models.generate_content`, `generate_content_stream`, `count_tokens` and `embed_content`, also under `client.aio`, from `genai.Client` for the Gemini API and Vertex AI. Gemini's automatic function calling runs your Python functions itself, so a replay runs them again: record them with `run.call_tool`, or turn it off. With Vertex and a project, a replay still asks Google for a sign-in token, so it needs your Google credentials, though no model is called.

Calls are recorded as they cross the wire, so `with_raw_response`, `.parse` and the stream helpers are recorded too, a replay serves the same bytes to whatever version of the package is installed, and the client keeps its proxies, timeouts and limits. Everything else on the client works as before. The SDK depends on neither package; it wraps the one the agent already uses.

**No code changes.** `helium run agent.py` starts the agent and records each launch as one run: every OpenAI, Anthropic, Google and LangChain call it makes. For one run per task instead, put `@helium.agent()` on the function that handles one task; `helium add-recording agent.py` shows that change as a diff:

```python
import kasanki_helium as helium


@helium.agent()
def handle_ticket(ticket): ...
```

To build a client yourself, pass it a recording HTTP client instead:

```python
with helium.run("support-agent", input=ticket) as run:
    client = OpenAI(http_client=run.http_client())
    # AsyncOpenAI(http_client=run.async_http_client())
```

Given a **LangChain** model (`langchain-openai` 1.5+ and `langchain-anthropic` 1.7.7+: `ChatOpenAI`, `AzureChatOpenAI` and models built on them, `ChatAnthropic`, `OpenAIEmbeddings`, or one of these after `bind_tools`), `run.wrap` returns a copy whose clients are wrapped, and leaves the original as it was. An agent holds its model, so build it with the copy inside the run:

```python
model = ChatOpenAI(model="gpt-5")

with helium.run("support-agent", input=ticket) as run:
    agent = create_agent(run.wrap(model), tools)
    result = agent.invoke({"messages": [...]})
```

The tools a LangChain agent runs are not recorded yet.

**Replay.** A recorded run can be run again with every input served from the recording: tools, clocks, random draws and model answers (rebuilt by the provider package from what was recorded, so nothing reaches the network). The engine says whether the agent did exactly the same, or where it diverged:

```python
# handle_ticket(ticket) runs the agent for one input
result = helium.replay(run_id, entry=handle_ticket)
result.matched, result.divergence

fork = helium.fork(
    run_id,
    at=14,
    entry=handle_ticket,
    change=helium.Change(system_prompt=prompt_v2),
    live={"send_email": 1},
)
fork.run_id  # a new run, branching from the original
```

`helium replay <run_id>` (or `python -m kasanki_helium replay <run_id>`) does the same from a terminal, using the entry point recorded with the run (the function that opened `helium.run`, when it takes the task input) or `[tool.kasanki-helium] entry` in `pyproject.toml`. After a fork point, tools marked `run.call_tool(..., tool_class="pure")` run again; others get their recorded answer or are blocked, unless listed in `live`.

**Graders.** A grader scores a run from 0 to 1 with a reason. Code graders are your own functions; judge graders are criteria the engine's AI analysis model answers yes, no or "does not apply" (switch AI analysis on first):

```python
# shop/graders.py, named in pyproject.toml: [tool.kasanki-helium] graders = "shop.graders"
@helium.grader
def one_refund(run: helium.GradedRun) -> helium.Grade:
    done = [c for c in run.tool_calls("refund_order") if c.error is None]
    return helium.Grade(1.0 if len(done) == 1 else 0.0, f"{len(done)} refunds")


follows_policy = helium.judge(
    "follows_policy",
    criteria=[
        "Never promises a refund the policy in the system prompt does not allow.",
    ],
)
```

A run too long for one request is read by the judge with read-only tools, choosing what each criterion needs; if it read only part of the run, the grade says how much ("the judge read 2,340 of 10,000 steps").

`helium grade <run_id>…` (or `helium.grade(run_id)`) grades recorded runs with them and stores each grade beside its run. A grader that raises gives no grade, never a 0. Exit code 0 every run passed, 1 one failed or could not be graded, 2 grading could not run.

**MCP servers** (`helium-solo[mcp]`). A server's tools as ordinary tools of the agent:

```python
shop = helium.mcp.connect(
    name="shop", command=["uvx", "shop-mcp"]
)  # or url=..., token_env="SHOP_TOKEN"
with helium.run("agent") as run:
    tools = shop.tools(run)  # for the model
    shop.call(run, "reserve", order_id="88231")  # recorded
```

A replay is served from the recording and starts no server. Tests use the sandbox's tool of that name, which can be a throwaway server (`test_server.as_sandbox()`). What a server offers is pinned in `helium/mcp.lock`: `helium mcp check` says if it changed, and `helium mcp pin` accepts a change you have read. A change is flagged and never stops an agent. Login is a token read from a variable you set; servers that only accept OAuth don't work yet.

**Searching documents.** A plain install searches by keyword; `helium-solo[search]` adds vectors, using a 67 MB model that runs on the CPU:

```python
docs = helium.search.open("policies/")  # Markdown and plain text
with helium.run("agent") as run:
    tools = [docs.tool()]  # for the model
    found = docs.search(run, "how long do refunds take")
```

Each search records its query, what it found and the index's hash. A replay is served from that, and a test reads a fixed index. In a grader, `helium.search.found(run, "refunds.txt")` says whether a search found that document. `helium index <folder> --query "..."` builds and tries it from the command line.

**Memory.** What an agent keeps between runs, in one readable file per scope:

```python
memory = helium.memory.open(
    "helium/memory", scope=customer_id
)  # your code sets the scope
with helium.run("agent") as run:
    tools = memory.tools()  # for the model
    memory.remember(run, "prefers email", key="contact")
    memory.recall(run, "contact")
```

The scope is never something the model can choose, which keeps customers apart. Searching is by keyword. `helium memory list`, `show` and `forget` work from the command line. **`forget` removes a fact from memory, not from past recordings**, which are tamper-evident, so plan for that if you may have to erase someone's data.

**A container you run.** `helium pack` writes a `Dockerfile`, compose file and a small HTTP service around the project's entry point, and starts nothing. `helium pack --check` builds it, runs a case inside with no keys, and looks for keys in every layer. Helium hosts nothing.

`helium.run(..., streaming=...)` sets what a streamed answer records: `"final_and_timings"` (the default: the final answer and when each piece arrived), `"every_chunk"` (also every piece) or `"final_only"`.

Secrets are always stripped before anything is recorded: API keys in headers or text, tokens, passwords in URLs.

The engine must be running on the same machine (`helium engine run`). The SDK finds it through the system-wide data folder, or the folder named by `KASANKI_HELIUM_DATA_DIR`.

## Development

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.10 for this project automatically.

```
uv sync
uv run ruff format --check
uv run ruff check
uv run mypy
uv run pytest
```

The end-to-end tests run against the real engine. Build it first (`cargo build` in `engine/`), or set `HELIUM_BIN` to a `helium` program; without either they are skipped. The D12 and W8 (model call) benchmarks run with `HELIUM_BENCHMARK=1`, ideally against a release build. `openai`, `anthropic` and the LangChain packages are development dependencies only, for the wrappers' tests; CI also runs those on the oldest supported versions (`uv run --with "openai==3.0.0" pytest tests/test_openai.py`, `anthropic==2.0.0` with `tests/test_anthropic.py`, and the versions in `.github/workflows/ci.yml` with `tests/test_langchain.py`).

**Checking a data folder yourself.** `python -m kasanki_helium.verify --data-dir <folder>` checks every run and checkpoint in a Helium data folder, with the folder's public key or one given with `--public-key`, and says what was changed, removed or added. It is plain Python with no dependencies, and gives the same answer as `helium verify`.

