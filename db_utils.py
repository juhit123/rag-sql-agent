from rag import add_document

def add_table(table_name: str, rows: list[dict]):
    """
    Stores a table in Chroma by saving each row as a document.
    """
    for idx, row in enumerate(rows, start=1):
        text = f"Table: {table_name}, Row {idx}: " + ", ".join(
            f"{k}={v}" for k, v in row.items()
        )
        add_document(f"{table_name}_{idx}", text, {"table": table_name})

