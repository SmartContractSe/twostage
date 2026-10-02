from __future__ import annotations
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import hashlib
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from scipy.special import iv
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
import torch
from tree_sitter import Language, Parser
import tree_sitter_solidity


@dataclass(frozen=True)
class ProNEConfig:
    dimension: int = 128
    step: int = 10
    mu: float = 0.2
    theta: float = 0.5
    random_state: int | None = 20260725


class SmartInspectProNE:
    def __init__(self, config: ProNEConfig):
        self.config = config

    @staticmethod
    def _normalized_embedding(
        left: np.ndarray, singular_values: np.ndarray
    ) -> np.ndarray:
        left = np.asarray(left) * np.sqrt(singular_values)
        return preprocessing.normalize(left, norm="l2")

    def _sparse_embedding(self, matrix: sp.spmatrix) -> np.ndarray:
        max_rank = min(matrix.shape)
        if max_rank == 0:
            return np.zeros((matrix.shape[0], 0), dtype=np.float64)
        dimension = min(self.config.dimension, max_rank)
        (left, singular_values, _) = randomized_svd(
            sp.csc_matrix(matrix),
            n_components=dimension,
            n_iter=5,
            random_state=self.config.random_state,
        )
        return self._normalized_embedding(left, singular_values)

    def _dense_embedding(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.zeros((matrix.shape[0], 0), dtype=np.float64)
        (left, singular_values, _) = linalg.svd(
            matrix, full_matrices=False, check_finite=False, overwrite_a=True
        )
        dimension = min(self.config.dimension, left.shape[1])
        return self._normalized_embedding(
            np.asarray(left[:, :dimension]), np.asarray(singular_values[:dimension])
        )

    def _pre_factorization(self, adjacency: sp.csr_matrix) -> np.ndarray:
        context = preprocessing.normalize(adjacency, norm="l1")
        negative_distribution = np.asarray(context.sum(axis=0))[0] ** 0.75
        negative_sum = float(negative_distribution.sum())
        if negative_sum <= 0.0:
            raise ValueError("ProNE graph has no usable edges")
        negative_distribution /= negative_sum
        negative = adjacency.dot(sp.diags(negative_distribution, format="csr"))
        context = context.tocsr(copy=True)
        negative = negative.tocsr(copy=True)
        context.data[context.data <= 0] = 1
        negative.data[negative.data <= 0] = 1
        context.data = np.log(context.data)
        negative.data = np.log(negative.data)
        return self._sparse_embedding(context - negative)

    def _spectral_propagation(
        self, adjacency: sp.csr_matrix, embedding: np.ndarray
    ) -> np.ndarray:
        if self.config.step == 1:
            return embedding
        node_count = adjacency.shape[0]
        adjacency_with_identity = sp.eye(node_count, format="csr") + adjacency
        normalized_adjacency = preprocessing.normalize(
            adjacency_with_identity, norm="l1"
        )
        laplacian = sp.eye(node_count, format="csr") - normalized_adjacency
        shifted = laplacian - self.config.mu * sp.eye(node_count, format="csr")
        lx0 = embedding
        lx1 = shifted.dot(embedding)
        lx1 = 0.5 * shifted.dot(lx1) - embedding
        convolution = iv(0, self.config.theta) * lx0
        convolution -= 2 * iv(1, self.config.theta) * lx1
        for order in range(2, self.config.step):
            lx2 = shifted.dot(lx1)
            lx2 = shifted.dot(lx2) - 2 * lx1 - lx0
            if order % 2 == 0:
                convolution += 2 * iv(order, self.config.theta) * lx2
            else:
                convolution -= 2 * iv(order, self.config.theta) * lx2
            lx0 = lx1
            lx1 = lx2
        propagated = adjacency_with_identity.dot(embedding - convolution)
        return self._dense_embedding(np.asarray(propagated))

    def fit_transform(self, adjacency: sp.spmatrix) -> np.ndarray:
        adjacency = sp.csr_matrix(adjacency, dtype=np.float64)
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("ProNE adjacency matrix must be square")
        initial = self._pre_factorization(adjacency)
        return self._spectral_propagation(adjacency, initial)


_WORD_RE = re.compile(b"^[A-Za-z_$][A-Za-z0-9_$]*$")


@dataclass(frozen=True)
class LexicalVocabulary:
    tokens: tuple[str, ...]
    token_bytes_to_index: dict[bytes, int]
    scanner: re.Pattern[bytes]
    words: int
    sha256: str

    @classmethod
    def from_tokens(cls, tokens: Sequence[str]) -> "LexicalVocabulary":
        ordered = tuple((str(token) for token in tokens))
        if not ordered:
            raise ValueError("lexical vocabulary must not be empty")
        if len(ordered) != len(set(ordered)):
            raise ValueError("lexical vocabulary contains duplicates")
        encoded = {
            token.encode("utf-8"): index for (index, token) in enumerate(ordered)
        }
        symbolic = sorted(
            (value for value in encoded if _WORD_RE.fullmatch(value) is None),
            key=lambda value: (-len(value), value),
        )
        if not symbolic:
            symbol_pattern = b"(?!)"
        else:
            symbol_pattern = (
                b"(?:" + b"|".join((re.escape(value) for value in symbolic)) + b")"
            )
        skip_pattern = b"//[^\\r\\n]*(?:\\r?\\n|\\Z)|/\\*(?:[^*]|\\*(?!/))*(?:\\*/|\\Z)|\"(?:\\\\[\\s\\S]|[^\"\\\\])*(?:\"|\\Z)|'(?:\\\\[\\s\\S]|[^'\\\\])*(?:'|\\Z)"
        scanner = re.compile(
            b"(?P<skip>"
            + skip_pattern
            + b")|(?P<word>[A-Za-z_$][A-Za-z0-9_$]*)|(?P<symbol>"
            + symbol_pattern
            + b")"
        )
        digest = hashlib.sha256(("\n".join(ordered) + "\n").encode("utf-8")).hexdigest()
        return cls(
            tokens=ordered,
            token_bytes_to_index=encoded,
            scanner=scanner,
            words=(len(ordered) + 63) // 64,
            sha256=digest,
        )


def _set_bit(signature: np.ndarray, index: int) -> None:
    signature[index // 64] |= np.uint64(1) << np.uint64(index % 64)


def signature_from_bytes(source: bytes, vocabulary: LexicalVocabulary) -> np.ndarray:
    signature = np.zeros(vocabulary.words, dtype=np.uint64)
    lookup = vocabulary.token_bytes_to_index
    for match in vocabulary.scanner.finditer(source):
        if match.lastgroup == "skip":
            continue
        index = lookup.get(match.group())
        if index is not None:
            _set_bit(signature, index)
    return signature


DEFAULT_PRONE_CONFIG = ProNEConfig(
    dimension=128, step=10, mu=0.2, theta=0.5, random_state=20260725
)


def build_function_token_matrix(
    signatures: np.ndarray, *, token_count: int, chunk_rows: int = 200000
) -> sp.csr_matrix:
    packed = np.asarray(signatures, dtype=np.uint64)
    if packed.ndim != 2:
        raise ValueError("signatures must be a two-dimensional uint64 array")
    if not 0 < token_count <= packed.shape[1] * 64:
        raise ValueError("token_count exceeds packed signature width")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    for start in range(0, packed.shape[0], chunk_rows):
        stop = min(start + chunk_rows, packed.shape[0])
        current = np.ascontiguousarray(packed[start:stop])
        bits = np.unpackbits(current.view(np.uint8), axis=1, bitorder="little")[
            :, :token_count
        ]
        (rows, columns) = np.nonzero(bits)
        row_parts.append(rows.astype(np.int64, copy=False) + start)
        column_parts.append(columns.astype(np.int32, copy=False))
    if row_parts:
        rows = np.concatenate(row_parts)
        columns = np.concatenate(column_parts)
    else:
        rows = np.empty(0, dtype=np.int64)
        columns = np.empty(0, dtype=np.int32)
    matrix = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, columns)),
        shape=(packed.shape[0], token_count),
        dtype=np.float32,
    ).tocsr()
    matrix.sort_indices()
    return matrix


def function_token_adjacency(matrix: sp.csr_matrix) -> sp.csr_matrix:
    features = sp.csr_matrix(matrix, dtype=np.float64)
    (function_count, token_count) = features.shape
    empty_functions = sp.csr_matrix((function_count, function_count), dtype=np.float64)
    empty_tokens = sp.csr_matrix((token_count, token_count), dtype=np.float64)
    return sp.bmat(
        [[empty_functions, features], [features.T, empty_tokens]],
        format="csr",
        dtype=np.float64,
    )


TOKENS = (
    "=",
    "function",
    "contract",
    "uint256",
    "address",
    "return",
    "returns",
    "uint",
    "public",
    "if",
    "bool",
    "==",
    "internal",
    "true",
    "+",
    ">",
    "!=",
    "string",
    "*",
    ">=",
    "<",
    "<=",
    "/",
    "pure",
    "bytes",
    "-",
    "&&",
    "else",
    "bytes32",
    "view",
    "emit",
    "++",
    "!",
    "+=",
    "memory",
    "false",
    "||",
    "-=",
    "for",
    "external",
    "payable",
    ":",
    "new",
    "uint8",
    "private",
    "uint64",
    "ether",
    "**",
    ":=",
    "storage",
    "?",
    "add",
    "var",
    "mload",
    "uint32",
    "assembly",
    "uint128",
    "%",
    "uint16",
    "delete",
    "int256",
    "int",
    "while",
    "byte",
    "mstore",
    "days",
    "let",
    "--",
    "&",
    "break",
    "bytes4",
    "uint160",
    "extcodesize",
    "*=",
    "sub",
    "<<",
    "finney",
    ">>",
    "/=",
    "and",
    "|",
    "div",
    "weeks",
    "^",
    "eq",
    "hours",
    "minutes",
    "mstore8",
    "continue",
    "years",
    "hex",
    "xor",
    "bytes8",
    "case",
    "uint40",
    "or",
    "mul",
    "|=",
    "switch",
    "not",
    "call",
    "number",
    "~",
    "timestamp",
    "lt",
    "sload",
    "bytes20",
    "bytes16",
    "iszero",
    "int8",
    "bytes1",
    "blockhash",
    "coinbase",
    "wei",
    "returndatasize",
    "bytes2",
    "uint24",
    "keccak256",
    "gas",
    "calldatasize",
    "uint96",
    "default",
    "revert",
    "exp",
    "int32",
    ">>=",
    "szabo",
    "gt",
    "uint80",
    "uint112",
    "fallback",
    "delegatecall",
    "bytes12",
    "uint120",
    "=>",
    "mapping",
    "returndatacopy",
    "seconds",
    "<<=",
    "calldatacopy",
    "int64",
    "calldata",
    "int128",
    "uint48",
    "^=",
    "sstore",
    "bytes14",
    "do",
    "int16",
    "uint88",
    "mod",
    "uint192",
    "uint176",
    "msize",
    "bytes5",
    "create",
    "bytes3",
    "bytes24",
    "uint224",
    "uint200",
    "&=",
    "bytes6",
    "uint240",
    "staticcall",
    "uint208",
    "invalid",
    "bytes13",
    "uint56",
    "%=",
    "bytes10",
    "calldataload",
    "extcodecopy",
    "caller",
    "uint152",
    "uint248",
    "bytes31",
    "callvalue",
    "bytes7",
    "uint72",
    "origin",
    "pop",
    "receive",
    "from",
    "log0",
    "log1",
    "log2",
    "log3",
    "log4",
    "sgt",
    "uint136",
    "modifier",
    "bytes15",
    "int104",
    "int112",
    "int120",
    "int136",
    "int144",
    "int152",
    "int160",
    "int168",
    "int176",
    "int184",
    "int192",
    "int200",
    "int208",
    "int216",
    "int224",
    "int232",
    "int24",
    "int240",
    "int248",
    "int40",
    "int48",
    "int56",
    "int72",
    "int80",
    "int88",
    "int96",
    "uint104",
    "uint144",
    "uint168",
    "uint184",
    "uint216",
    "uint232",
    "balance",
    "bytes22",
    "bytes23",
    "bytes28",
    "gaslimit",
    "indexed",
    "slt",
    "bytes18",
    "event",
)
VOCABULARY = LexicalVocabulary.from_tokens(TOKENS)
OPS = {"A": (0.3, 0.2), "B": (0.7, 0.2), "C": (0.3, 0.8)}
FUNCTION_TYPES = {
    "function_definition",
    "constructor_definition",
    "modifier_definition",
    "fallback_receive_definition",
}


def graph_similarities(seeds, targets):
    if not targets or not seeds or any((not sources for sources in seeds.values())):
        raise ValueError("Nonempty targets and seed categories are required")

    def features(sources):
        signatures = np.stack(
            [signature_from_bytes(s.encode("utf-8"), VOCABULARY) for s in sources]
        )
        return build_function_token_matrix(signatures, token_count=len(TOKENS))

    target_matrix = features(targets)
    active = np.flatnonzero(np.asarray(target_matrix.getnnz(axis=1)).reshape(-1) > 0)
    if not active.size:
        raise ValueError("All target functions have empty token signatures")
    seed_matrices = {
        category: features(sources) for (category, sources) in seeds.items()
    }
    for matrix in seed_matrices.values():
        if np.any(np.asarray(matrix.getnnz(axis=1)).reshape(-1) == 0):
            raise ValueError("Seed has no vocabulary tokens")
    combined = sp.vstack([target_matrix[active], *seed_matrices.values()], format="csr")
    adjacency = function_token_adjacency(combined)
    embedding = SmartInspectProNE(DEFAULT_PRONE_CONFIG).fit_transform(adjacency)
    embedding = preprocessing.normalize(
        embedding[: combined.shape[0]], norm="l2", copy=False
    )
    scores = {}
    offset = len(active)
    for category, matrix in seed_matrices.items():
        values = np.asarray(
            embedding[: len(active)] @ embedding[offset : offset + matrix.shape[0]].T,
            dtype=np.float32,
        )
        complete = np.full((len(targets), matrix.shape[0]), -1.0, dtype=np.float32)
        complete[active] = np.where(np.isfinite(values), values, -1.0)
        scores[category] = complete
        offset += matrix.shape[0]
    return scores


@dataclass(frozen=True)
class FunctionAST:
    node_types: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]

    @property
    def node_count(self):
        return len(self.node_types)


def parse_function(source):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        language = Language(tree_sitter_solidity.language())
    parser = Parser(language)

    def select(root):
        (pending, functions) = ([root], [])
        while pending:
            node = pending.pop()
            if node.type in FUNCTION_TYPES:
                functions.append(node)
            pending.extend(reversed(node.children))
        functions.sort(key=lambda node: (node.start_byte, node.end_byte))
        return (
            max(functions, key=lambda node: node.end_byte - node.start_byte)
            if functions
            else None
        )

    data = source.encode("utf-8")
    tree = parser.parse(data)
    root = select(tree.root_node)
    if root is None or root.has_error:
        tree = parser.parse(
            b"contract __SmartInspectFunctionFragment {\n" + data + b"\n}\n"
        )
        root = select(tree.root_node)
    if root is None or root.has_error:
        raise ValueError("Expected a valid complete Solidity function")
    (types, edges) = ([], [])

    def visit(node):
        index = len(types)
        types.append(str(node.type))
        for child in node.children:
            if child.type != "comment":
                child_index = visit(child)
                edges.append((index, child_index))
        return index

    visit(root)
    return FunctionAST(tuple(types), tuple(edges))


def exact_type_score(seed, target, device="cuda:0"):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ExactType requires CUDA")
    reversed_kernel = target.node_count < seed.node_count
    (kernel, search) = (target, seed) if reversed_kernel else (seed, target)
    result = {
        "score": 0.0,
        "status": "empty_edge_set",
        "kernel_source": "target" if reversed_kernel else "seed",
        "kernel_edge_entries": 2 * len(kernel.edges),
        "row_shift": None,
        "column_shift": None,
    }
    if not kernel.edges or not search.edges:
        return result
    vocabulary = {
        name: i
        for (i, name) in enumerate(sorted(set(kernel.node_types + search.node_types)))
    }

    def tensors(graph):
        edges = torch.tensor(graph.edges, dtype=torch.long, device=device).T
        rows = torch.cat((edges[0], edges[1]))
        cols = torch.cat((edges[1], edges[0]))
        types = torch.tensor(
            [vocabulary[name] for name in graph.node_types],
            dtype=torch.long,
            device=device,
        )
        return (rows, cols, types)

    (kr, kc, kt) = tensors(kernel)
    (tr, tc, tt) = tensors(search)
    width = search.node_count - kernel.node_count + 1
    counts = torch.zeros(width * width, dtype=torch.long, device=device)
    chunk = max(1, 2000000 // tr.numel())
    aligned = False
    for start in range(0, kr.numel(), chunk):
        (rows, cols) = (kr[start : start + chunk], kc[start : start + chunk])
        (dr, dc) = (tr[None, :] - rows[:, None], tc[None, :] - cols[:, None])
        valid = (dr >= 0) & (dr < width) & (dc >= 0) & (dc < width)
        aligned = bool(valid.any()) or aligned
        typed = (
            valid
            & (kt[rows, None] == tt[tr][None, :])
            & (kt[cols, None] == tt[tc][None, :])
        )
        counts += torch.bincount((dr * width + dc)[typed], minlength=width * width)
    if not aligned:
        result["status"] = "no_valid_alignment"
        return result
    best = int(counts.argmax())
    result.update(
        score=float(counts[best]) / kr.numel(),
        status="ok_target_kernel" if reversed_kernel else "ok",
        row_shift=best // width,
        column_shift=best % width,
    )
    return result


def detect(seeds, targets, op="A", device="cuda:0"):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SmartInspect requires CUDA")
    (graph_threshold, ast_threshold) = OPS[op]
    similarities = graph_similarities(seeds, targets)
    (target_asts, seed_asts, results) = ({}, {}, {})
    for category, scores in similarities.items():
        rows = [
            {"target_index": i, "detected": False, "pairs": []}
            for i in range(len(targets))
        ]
        for i, j in zip(*np.nonzero(np.isfinite(scores) & (scores > graph_threshold))):
            (i, j) = (int(i), int(j))
            if i not in target_asts:
                target_asts[i] = parse_function(targets[i])
            key = (category, j)
            if key not in seed_asts:
                seed_asts[key] = parse_function(seeds[category][j])
            ast_score = exact_type_score(seed_asts[key], target_asts[i], device)
            passed = (
                ast_score["status"] in {"ok", "ok_target_kernel"}
                and ast_score["score"] > ast_threshold
            )
            rows[i]["pairs"].append(
                {
                    "seed_index": j,
                    "graph_score": float(scores[i, j]),
                    "ast": ast_score,
                    "passed": passed,
                }
            )
            rows[i]["detected"] = rows[i]["detected"] or passed
        results[category] = rows
    return results


EXAMPLE = 'function withdraw(uint256 amount) public {\n    require(balances[msg.sender] >= amount);\n    (bool ok, ) = msg.sender.call.value(amount)("");\n    require(ok);\n    balances[msg.sender] -= amount;\n}'


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--seed", type=Path, action="append")
    cli.add_argument("--target", type=Path, action="append")
    cli.add_argument("--op", choices=OPS, default="A")
    cli.add_argument("--device", default="cuda:0")
    args = cli.parse_args()
    if bool(args.seed) != bool(args.target):
        cli.error("--seed and --target must be supplied together")
    torch.set_num_threads(2)
    seeds = (
        [p.read_text(encoding="utf-8") for p in args.seed] if args.seed else [EXAMPLE]
    )
    targets = (
        [p.read_text(encoding="utf-8") for p in args.target]
        if args.target
        else [EXAMPLE.replace("amount", "quantity")]
    )
    result = detect({"query": seeds}, targets, op=args.op, device=args.device)
    if not args.seed and (not result["query"][0]["detected"]):
        raise RuntimeError("Renamed function did not match")
    print(
        json.dumps(
            {
                "status": "ok",
                "device": str(torch.device(args.device)),
                "vocabulary": len(TOKENS),
                "op": args.op,
                "results": result,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
