import argparse
import sqlite3
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql_path", required=True, help="This is the path the SQL database that maps keys to smiles and DOCK scores")
    parser.add_argument("--fraction_for_best", required=True, type=float, help="Fraction of nodes to consider as top scoring")
    args = parser.parse_args()

    if not os.path.exists(args.sql_path):
        raise ValueError("SQL Database path does not exist")

    if args.fraction_for_best <= 0 or args.fraction_for_best > 1:
        raise ValueError("fraction_for_best must be between 0 and 1")

    conn = sqlite3.connect(args.sql_path)
    c = conn.cursor()

    # Retrieve the lowest fraction of scores
    query = f"""
        WITH ordered_nodes AS (
            SELECT node_key, score,
                   ROW_NUMBER() OVER (ORDER BY score ASC) AS row_num,
                   COUNT(*) OVER () AS total_count
            FROM nodes
        )
        SELECT node_key, score
        FROM ordered_nodes
        WHERE row_num <= total_count * ?
    """
    c.execute(query, (args.fraction_for_best,))
    results = c.fetchall()
    conn.close()

    with open(f"{os.path.dirname(args.sql_path)}/best_nodes", "w") as f:
        for key, score in results:
            f.write(f"{key} {score}\n")
