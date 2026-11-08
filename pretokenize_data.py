"""Entrypoint for DCLM dataset pre-tokenization, sharding, and worker partition generation."""

from data.preprocess import (
    build_preprocess_arg_parser,
    init_worker,
    parse_preprocess_args,
    run_preprocessing as main,
    tokenize_doc,
)


__all__ = [
    "init_worker",
    "tokenize_doc",
    "main",
    "build_preprocess_arg_parser",
    "parse_preprocess_args",
]


if __name__ == "__main__":
    args = parse_preprocess_args()
    main(args)


