import os
import time

import chromadb


def main() -> None:
    host = os.getenv("MARINE_RAG_CHROMA_HOST")
    if not host:
        return

    port = int(os.getenv("MARINE_RAG_CHROMA_PORT", "8000"))
    ssl = os.getenv("MARINE_RAG_CHROMA_SSL", "false").lower() == "true"
    timeout_seconds = int(os.getenv("MARINE_RAG_CHROMA_WAIT_SECONDS", "60"))
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            client = chromadb.HttpClient(host=host, port=port, ssl=ssl)
            client.heartbeat()
            print(f"Chroma is ready at {host}:{port}")
            return
        except Exception as exc:
            last_error = exc
            print(f"Waiting for Chroma at {host}:{port}...")
            time.sleep(2)

    raise RuntimeError(f"Chroma did not become ready within {timeout_seconds}s: {last_error}")


if __name__ == "__main__":
    main()

