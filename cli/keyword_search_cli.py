#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    bm25_idf_command,
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Count term frequencies")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to count")

    idf_parser = subparsers.add_parser(
        "idf", help="Calculate inverse document frequency"
    )
    idf_parser.add_argument("term", type=str, help="Term to calculate IDF for")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Calculate term frequency-inverse document frequency"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to calculate TF-IDF for")

    bm25idf_parser = subparsers.add_parser(
        "bm25idf", help="Calculate BM25 inverse document frequency"
    )
    bm25idf_parser.add_argument("term", type=str, help="Term to calculate BM25 IDF for")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "tf":
            print("Getting term frequency...")
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document {args.doc_id}: {tf}")

        case "idf":
            print("Calculating inverse document frequency...")
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            print("Calculating term frequency-inverse document frequency...")
            tfidf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF of '{args.term}' in document {args.doc_id}: {tfidf:.2f}")

        case "bm25idf":
            print("Calculating BM25 score...")
            bm25 = bm25_idf_command(args.term)
            print(f"BM25 score of '{args.term}': {bm25:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
