# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "src/automaton.cpp":
    cdef cppclass DraftBufferCPP "DraftBuffer":
        vector[int] tree_candidates
        vector[int] position_ids
        vector[vector[int]] candidates
        vector[vector[int]] attn_mask
        vector[vector[int]] retrieve_indices

    cdef cppclass AutomatonCPP "Automaton":
        AutomatonCPP(int max_nodes)
        void init_logits(int vocab_size, int top_k) except +
        void update(const vector[int]& tokens, const vector[vector[int]]& adj_vectors) except +
        void insert(const vector[int]& pattern, int freq) except +
        void build() except +
        void trans_tokens(const vector[int]& tokens) except +
        DraftBufferCPP retrieve(int next_token, int max_num_draft, bool is_chain) except +

# --- DraftBuffer Python Wrapper ---

cdef class DraftBuffer:
    cdef public list tree_candidates
    cdef public list position_ids
    cdef public list candidates
    cdef public list attn_mask
    cdef public list retrieve_indices

    def __init__(self):
        pass

    @staticmethod
    cdef DraftBuffer from_cpp(DraftBufferCPP cppbuf):
        cdef DraftBuffer buf = DraftBuffer()
        buf.tree_candidates = cppbuf.tree_candidates
        buf.position_ids = cppbuf.position_ids
        buf.candidates = [[item for item in inner] for inner in cppbuf.candidates]
        buf.attn_mask = [[item for item in inner] for inner in cppbuf.attn_mask]
        buf.retrieve_indices = [[item for item in inner] for inner in cppbuf.retrieve_indices]
        return buf

    def __repr__(self):
        return (f"<DraftBuffer tree_candidates={self.tree_candidates}, position_ids={self.position_ids}, "
                f"candidates={self.candidates}, attn_mask={self.attn_mask}, retrieve_indices={self.retrieve_indices}>")

# --- Automaton Python Interface ---

cdef class Automaton:
    cdef AutomatonCPP* _cpp

    def __cinit__(self, max_nodes=1000):
        self._cpp = new AutomatonCPP(max_nodes)

    def __dealloc__(self):
        del self._cpp

    def init_logits(self, vocab_size, top_k):
        self._cpp.init_logits(vocab_size, top_k)

    def update(self, tokens, adj_vectors):
        self._cpp.update(tokens, adj_vectors)

    def insert(self, pattern, freq=1):
        self._cpp.insert(pattern, freq)

    def build(self):
        self._cpp.build()

    def trans_tokens(self, tokens):
        self._cpp.trans_tokens(tokens)

    def retrieve(self, next_token, max_num_draft=64, is_chain=False):
        cdef DraftBufferCPP cppbuf = self._cpp.retrieve(<int>next_token, max_num_draft, is_chain)
        return DraftBuffer.from_cpp(cppbuf)