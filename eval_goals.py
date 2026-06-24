"""
eval_goals.py - the 40-goal test set Atlas is scored against.

Varied on purpose so the score reflects general research ability rather than one
kind of question: comparisons, definitions, how-it-works, tradeoffs, recent state
(2026), and specific factual lookups across several domains.
"""

GOALS = [
    # --- Comparisons (8) ---
    "Compare LangGraph and CrewAI for building multi-agent systems.",
    "Compare retrieval-augmented generation (RAG) and fine-tuning for adapting an LLM to a domain.",
    "Compare PostgreSQL and MongoDB for a high-write web application.",
    "Compare solar and wind energy on cost and reliability as of 2026.",
    "Compare React and Svelte for building a modern web frontend.",
    "Compare REST and GraphQL for designing a public API.",
    "Compare Rust and Go for building high-performance backend services.",
    "Compare battery-electric and hydrogen fuel-cell vehicles for long-haul trucking.",

    # --- Definitions / explanations (8) ---
    "What is retrieval-augmented generation (RAG) and how does it work?",
    "What is the transformer architecture in deep learning?",
    "What is a zero-knowledge proof and where is it used?",
    "What is CRISPR gene editing and how does it work?",
    "What is the supervisor pattern in multi-agent LLM systems?",
    "What is quantum entanglement, explained simply?",
    "What is a vector database and why do AI applications use one?",
    "What is the difference between machine learning and deep learning?",

    # --- How-to / mechanisms (6) ---
    "How does HTTPS keep web traffic secure?",
    "How is a large language model trained, end to end?",
    "How does Bitcoin's proof-of-work consensus work?",
    "How does mRNA vaccine technology work?",
    "How does a self-driving car perceive its environment?",
    "How does DNS resolve a domain name to an IP address?",

    # --- Tradeoffs / should-I (6) ---
    "What are the tradeoffs of a microservices architecture versus a monolith?",
    "What are the pros and cons of remote work for software engineering teams?",
    "What are the risks and benefits of using LLMs for medical advice?",
    "Is nuclear power a good option for reducing carbon emissions?",
    "What are the tradeoffs of serverless computing for an early-stage startup?",
    "Should small businesses adopt AI chatbots for customer support?",

    # --- Recent / 2026 state (6) ---
    "What is the current state of AI agent frameworks in 2026?",
    "What are the latest advances in battery technology as of 2026?",
    "What major AI regulations exist or are emerging in 2026?",
    "What is the state of commercial fusion energy research in 2026?",
    "What are the most in-demand programming skills in 2026?",
    "What are the latest developments in reusable rocket technology?",

    # --- Factual / specific (6) ---
    "What were the main causes of the 2008 global financial crisis?",
    "What is the James Webb Space Telescope and what has it discovered?",
    "Who are the leading companies in the global EV market and what are their shares?",
    "What does research say about the health effects of intermittent fasting?",
    "What is the Mediterranean diet and what are its evidence-based benefits?",
    "What caused the decline of the Western Roman Empire, according to historians?",
]

assert len(GOALS) == 40, f"expected 40 goals, found {len(GOALS)}"
