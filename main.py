from src.rag_pipeline import QuantumRAGPipeline


def run_cli():
    print("=" * 60)
    print(" Quantum RAG System (Local, Hybrid Retrieval)")
    print("Type 'exit' to quit")
    print("=" * 60)

    rag = QuantumRAGPipeline()
    rag.ingest()

    while True:
        query = input("\nAsk: ").strip()

        if query.lower() in ["exit", "quit"]:
            print(" Exiting...")
            break

        if not query:
            continue

        response = rag.generate(query)

        print("\n Answer:")
        print(response.answer)

        print("\n Retrieved Contexts:")
        for i, rc in enumerate(response.retrieved_chunks, 1):
            print(f"\n--- Context {i} ---")
            print(f"Doc: {rc.chunk.doc_id}")
            print(f"Score: {rc.score:.3f}")
            print(f"Text: {rc.chunk.text[:200]}...")



def run_single_query(query):
    rag = QuantumRAGPipeline()
    rag.ingest()

    response = rag.generate(query)

    print("\nQuestion:", query)
    print("\nAnswer:", response.answer)

    print("\nRetrieved:")
    for rc in response.retrieved_chunks:
        print(f"[{rc.score:.3f}] {rc.chunk.doc_id}")


if __name__ == "__main__":
    run_cli()
