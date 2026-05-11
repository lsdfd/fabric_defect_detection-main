import json
import pathlib
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python scripts/run_notebook_cells.py <notebook_path> <output_log>")
        return 2

    nb_path = pathlib.Path(sys.argv[1]).resolve()
    log_path = pathlib.Path(sys.argv[2]).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(nb_path, as_version=4)

    with log_path.open("w", encoding="utf-8") as log:
        client = NotebookClient(
            nb,
            timeout=None,
            kernel_name="python3",
            resources={"metadata": {"path": str(nb_path.parent)}},
        )

        try:
            with client.setup_kernel():
                for i, cell in enumerate(nb["cells"]):
                    if cell.get("cell_type") != "code":
                        continue

                    src = "".join(cell.get("source", []))
                    preview_lines = src.strip().splitlines()
                    preview = preview_lines[0][:160] if preview_lines else "<empty>"
                    log.write(f"\n=== CELL {i}: {preview} ===\n")
                    log.flush()

                    try:
                        client.execute_cell(cell, i)
                    except CellExecutionError as exc:
                        log.write(f"CELL {i} ERROR: {exc}\n")
                        outputs = cell.get("outputs", [])
                        for out in outputs:
                            if "text" in out:
                                log.write("".join(out["text"]))
                            elif out.get("output_type") == "stream":
                                log.write(out.get("text", ""))
                            elif out.get("output_type") == "error":
                                log.write("\n".join(out.get("traceback", [])) + "\n")
                        raise

                    for out in cell.get("outputs", []):
                        if "text" in out:
                            log.write("".join(out["text"]))
                        elif out.get("output_type") == "stream":
                            log.write(out.get("text", ""))
                        elif out.get("output_type") == "error":
                            log.write("\n".join(out.get("traceback", [])) + "\n")
                        elif "data" in out and "text/plain" in out["data"]:
                            data = out["data"]["text/plain"]
                            if isinstance(data, list):
                                log.write("".join(data))
                            else:
                                log.write(str(data))
                            log.write("\n")
                    log.flush()
        finally:
            try:
                client._cleanup_kernel()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
