#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include <queue>
#include <algorithm>
#include <list>
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <memory>

struct TrieNode {
    std::unordered_map<int, TrieNode*> children;
    TrieNode* fail = nullptr;
    TrieNode* parent = nullptr;
    int token = -1;
    int freq = 0;
    int depth = 0;

    void clear() {
        parent = nullptr;
        fail = nullptr;
        children.clear();
        token = -1;
        freq = 0;
        depth = 0;
    }
};

struct DraftBuffer {
    std::vector<int> tree_candidates;
    std::vector<int> position_ids;

    std::vector<std::vector<int>> candidates;
    std::vector<std::vector<int>> attn_mask;
    std::vector<std::vector<int>> retrieve_indices;
};

class TokenBin {
private:
    std::vector<std::vector<int>> adj_matrix;
    int top_k;

    struct Node {
        int token;
        int pos_parent;
        int breadth;
        int depth;

        Node(int token, int pos_parent, int breadth, int depth)
        : token(token), pos_parent(pos_parent), breadth(breadth), depth(depth) {

        }
    };

public:
    TokenBin(int vocab_size, int top_k) : top_k(top_k) {
        adj_matrix.resize(vocab_size);
        for (auto& adj_vec : adj_matrix) {
            adj_vec.resize(top_k);
        }
        adj_matrix.shrink_to_fit();
    }

    void update(const std::vector<int>& input_ids, const std::vector<std::vector<int>>& adj_vectors) {
        for (size_t i = 0; i < input_ids.size(); ++i) {
            int token_id = input_ids[i];
            
            if (token_id >= adj_matrix.size()) {
                continue;
            }

            adj_matrix[token_id] = adj_vectors[i];
        }
    }

    std::vector<std::vector<int>> retrieve(int next_token, int max_num_draft, bool is_chain = false) {
        if (next_token >= adj_matrix.size()) {
            return {{next_token}};
        }

        std::vector<std::vector<int>> candidates;

        std::vector<Node> q;

        q.emplace_back(next_token, -1, is_chain ? 1 : top_k, 0);
        max_num_draft--;

        size_t head = 0;

        while (head < q.size()) {
            auto u = q[head++];

            int pos_u = head - 1; 
            int breadth = u.breadth, depth = u.depth, token = u.token;

            if (max_num_draft > 0 && breadth > 0 && token < adj_matrix.size()) {
                int next_breadth = depth == 1 ? breadth : (breadth >> 1);
                int next_depth = depth + 1;
                for (int i = 0; i < breadth && max_num_draft > 0; i++) {
                    int child = adj_matrix[token][i];
                    q.emplace_back(child, pos_u, std::max(1, next_breadth), next_depth);
                    next_breadth >>= 1;
                    max_num_draft--;
                }
            } else { // Leaf node
                std::vector<int> candidate;

                while (pos_u >= 0) {
                    candidate.push_back(q[pos_u].token);
                    pos_u = q[pos_u].pos_parent;
                }

                // leaf to root -> root to leaf
                candidates.emplace_back(candidate.rbegin(), candidate.rend());
            }
        }

        return candidates;
    }
};

class Trie {
protected:
    TrieNode* root = nullptr;
    std::vector<TrieNode> nodes;
    std::list<TrieNode*> lru_list;
    std::unordered_map<TrieNode*, std::list<TrieNode*>::iterator> lru_map;
    TrieNode* _cur_state;
    int _node_count;

    void touch(TrieNode* node) {
        auto it = lru_map.find(node);
        assert(it != lru_map.end());
        lru_list.splice(lru_list.begin(), lru_list, it->second);
        lru_map[node] = lru_list.begin();
    }

    void touch_prefix(TrieNode* node) {
        while (node) {
            if (root == nullptr) {
                node->fail = node; // Root's fail points to itself
            } else {
                node->fail = root; // All other nodes' fail points to root initially
            }
            auto it = lru_map.find(node);
            assert(it != lru_map.end());
            lru_list.splice(lru_list.begin(), lru_list, it->second);
            lru_map[node] = lru_list.begin();
            node = node->parent;
        }
    }

    TrieNode* get_new_node() {
        // Ensure the last node in the LRU list is empty
        assert(lru_list.back()->children.empty());
        TrieNode* node = lru_list.back();
        if (node->parent) {
            auto it = node->parent->children.find(node->token);
            if (it != node->parent->children.end() && it->second == node) {
                node->parent->children.erase(it); // Remove this node from parent's children
            }
        }
        node->clear();
        if (root == nullptr) {
            node->fail = node; // Root's fail points to itself
        } else {
            node->fail = root; // All other nodes' fail points to root initially
        }
        touch(node);
        _node_count++;
        return node; 
    }

public:
    Trie(int max_nodes) : nodes(max_nodes) {
        max_nodes = std::max(max_nodes, 1);
        _node_count = 0;
        for (auto& node : nodes) {
            node.clear();
            lru_list.push_back(&node);
            lru_map[&node] = prev(lru_list.end());
        }
        root = get_new_node();
        _cur_state = root;
    }

    int node_count() const {
        return std::min(_node_count, static_cast<int>(nodes.size()));
    }

    void insert(const std::vector<int>& pattern, int freq = 1) {
        TrieNode* u = root;
        u->freq += freq;
        for (int token : pattern) {
            touch(u); // Touch the current node
            if (!u->children.count(token)) {
                TrieNode* new_node = get_new_node();
                new_node->parent = u;
                new_node->token = token;
                new_node->depth = u->depth + 1;
                u->children[token] = new_node;
            }
            u = u->children[token];
            u->freq += freq;
        }
        touch(u); // Touch the leaf node
    }

    void reset(TrieNode* new_state = nullptr) {
        if (new_state == nullptr) {
            new_state = root; // Reset to root if no state is provided
        }
        if (!lru_map.count(new_state) || new_state != root && new_state->token == -1) {
            throw std::runtime_error("Not a valid state");
        }
        _cur_state = new_state;
        touch(_cur_state);
    }

    DraftBuffer flatten() {
        // Initialize the buffer
        DraftBuffer buf;

        int trie_size = node_count() - 1; // Without root

        buf.attn_mask.resize(trie_size);

        for (int i = 0; i < trie_size; i++) {
            buf.attn_mask[i].resize(trie_size);
        }

        std::queue<TrieNode*> q; // Queue for BFS

        std::map<TrieNode*, int> seq_pos;

        int visited = 0;

        for (const auto& [_, child] : root->children) {
            q.push(child);
        }

        while (!q.empty()) {
            auto u = q.front();

            q.pop();

            seq_pos[u] = visited++; // Assign position in BFS sequence

            auto parent = u->parent;

            auto pos_u = visited - 1, pos_parent = seq_pos[parent]; 

            buf.position_ids.push_back(u->depth - 1);
            buf.tree_candidates.push_back(u->token);

            if (parent != root) {
                std::copy(buf.attn_mask[pos_parent].begin(), buf.attn_mask[pos_parent].end(), buf.attn_mask[pos_u].begin());
            }

            buf.attn_mask[pos_u][pos_u] = 1;

            for (const auto& [_, child] : u->children) {
                q.emplace(child);
            }

            if (u->children.empty()) { // Leaf node
                std::vector<int> candidate;
                std::vector<int> indices;

                while (u != root) {
                    candidate.push_back(u->token);
                    indices.push_back(seq_pos[u]);
                    u = u->parent;
                }

                // leaf to root -> root to leaf
                buf.candidates.emplace_back(candidate.rbegin(), candidate.rend());
                buf.retrieve_indices.emplace_back(indices.rbegin(), indices.rend());
            }
        }

        return buf;
    }
};

class Automaton : public Trie {
private:
    int min_depth;
    std::unique_ptr<TokenBin> token_bin = nullptr;

public:
    Automaton(int max_nodes, int min_depth = 2) : Trie(max_nodes), min_depth(min_depth) {
        min_depth = std::max(min_depth, 1);
    }

    void init_logits(int vocab_size, int top_k) {
        token_bin = std::make_unique<TokenBin>(vocab_size, top_k);
    }

    void update(const std::vector<int>& input_ids, const std::vector<std::vector<int>>& adj_vectors) {
        if (token_bin) {
            token_bin->update(input_ids, adj_vectors);
        }
    }

    void build() {
        std::queue<TrieNode*> q;
        for (const auto& [_, child] : root->children) {
            child->fail = root;
            q.push(child);
        }
        while (!q.empty()) {
            TrieNode* cur = q.front();
            q.pop();
            for (const auto& [token, child] : cur->children) {
                TrieNode* f = cur->fail;
                while (f != root && !f->children.count(token)) {
                    f = f->fail;
                }
                if (f->children.count(token)) {
                    child->fail = f->children[token];
                } else {
                    child->fail = root;
                }
                q.push(child);
            }
        }
    }

    void trans_tokens(const std::vector<int>& tokens) {
        auto& u = _cur_state;
        for (const auto& token : tokens) {
            touch(u); // Touch the current node
            if (!u->children.count(token)) { // Might switch to another sub-Trie
                while (u != root && !u->children.count(token)) {
                    u = u->fail; // Keep going up the trie until we find a match or reach the root
                }
                touch_prefix(u); // Update the prefix after fail transition
            }
            if (u->children.count(token)) { // Otherwise we reach the root
                u = u->children[token];
            }
        }
        touch(u); // Touch the final state after processing all tokens
    }

    DraftBuffer retrieve(int next_token, int max_num_draft, bool is_chain = false) {
        if (max_num_draft <= 0) {
            throw std::invalid_argument("max_num_draft must be greater than 0");
        }

        auto u = _cur_state;

        std::vector<TrieNode*> borders;

        bool state_updated = false;

        // Get borders
        while (u != root) {
            if (u->children.count(next_token)) {
                auto v = u->children[next_token];
                if (v->depth >= min_depth && (borders.empty() || !is_chain)) {
                    borders.push_back(v);
                }
                if (!state_updated) {
                    _cur_state = v;
                    state_updated = true;
                }
            }
            u = u->fail; // Backtrack to the fail state
        }

        if (root->children.count(next_token)) {
            auto v = root->children[next_token];
            if (v->depth >= min_depth && (borders.empty() || !is_chain)) {
                borders.push_back(v);
            }
            if (!state_updated) {
                _cur_state = v;
                state_updated = true;
            }
        }

        for (auto node : borders) {
            touch_prefix(node); // Touch the border nodes
        }

        std::nth_element(borders.begin(), borders.begin() + std::min(max_num_draft, static_cast<int>(borders.size())), borders.end(),
            [this](TrieNode* a, TrieNode* b) { return a->freq > b->freq; }); // Sort borders based on frequency (decending order)

        // ((-freq, depth), (u, start_u))
        std::priority_queue<std::pair<std::pair<int, int>, std::pair<TrieNode*, TrieNode*>>> top_k; // Min-heap to keep track of the top_k nodes based on frequency
        
        std::queue<std::pair<TrieNode*, TrieNode*>> q; // (u, start_u) (of the sub-trie)

        for (const auto& border : borders) {
            q.emplace(border, border);
        }

        std::vector<std::pair<TrieNode*, TrieNode*>> current_layer; // (u, start_u)
        int current_depth = 0;

        // First BFS to find the top-k nodes based on frequency
        while (!q.empty()) {
            auto u = q.front().first, start_u = q.front().second;

            q.pop();

            auto depth = u->depth - start_u->depth;

            if (depth > current_depth) {
                bool updated = false;
                for (const auto& [v, start_v] : current_layer) {
                    if (top_k.size() < max_num_draft || v->freq > -top_k.top().first.first) {
                        updated = true;
                        top_k.emplace(std::make_pair(-v->freq, v->depth), std::make_pair(v, start_v));
                        if (top_k.size() > max_num_draft) {
                            top_k.pop(); // Maintain the size of the heap
                        }
                    }
                }
                current_layer.clear();
                current_depth = depth;
                if (!updated) {
                    break; // Note that the freq is non-increasing, so if not updated, break the BFS
                }
            }

            current_layer.emplace_back(u, start_u);

            if (!is_chain) {
                for (const auto& [_, child] : u->children) {
                    q.emplace(child, start_u);
                }
            } else {
                int max_child_freq = -1;
                TrieNode* max_child = nullptr;
                for (const auto& [_, child] : u->children) {
                    if (child->freq > max_child_freq) {
                        max_child_freq = child->freq;
                        max_child = child;
                    }
                }
                if (max_child) {
                    q.emplace(max_child, start_u);
                }
            }
        }

        // Process the last layer
        for (const auto& [v, start_v] : current_layer) {
            if (top_k.size() < max_num_draft || v->freq > -top_k.top().first.first) {
                top_k.emplace(std::make_pair(-v->freq, v->depth), std::make_pair(v, start_v));
                if (top_k.size() > max_num_draft) {
                    top_k.pop(); // Maintain the size of the heap
                }
            }
        }

        std::set<std::pair<TrieNode*, TrieNode*>> selected; // (u, start_u)

        while (!top_k.empty()) {
            auto [_, pair] = top_k.top();
            top_k.pop();
            selected.insert(pair);
        }
        
        // Including an empty node for root
        Trie candidate_trie(max_num_draft + 1);

        std::vector<int> candidate;
        candidate.push_back(next_token); // In case no border is selected

        for (auto [u, start_u] : selected) {
            bool is_candidate_leaf = true;

            for (const auto& [_, child] : u->children) {
                if (selected.count({child, start_u}) > 0) {
                    is_candidate_leaf = false;
                }
            }

            if (is_candidate_leaf) { // Leaf node
                candidate.clear();

                while (u != start_u) {
                    candidate.push_back(u->token);
                    u = u->parent;
                }

                // Current root
                candidate.push_back(u->token);
                std::reverse(candidate.begin(), candidate.end()); // Reverse to get the correct order
                candidate_trie.insert(candidate);
            }
        }

        if (token_bin) {
            // Same root, node count + 1
            auto aux_candidates = token_bin->retrieve(is_chain ? candidate.back() : next_token, max_num_draft - selected.size(), is_chain);
            for (auto aux_candidate : aux_candidates) {
                if (is_chain) {
                    aux_candidate.insert(aux_candidate.begin(), candidate.begin(), candidate.end() - 1);
                }
                candidate_trie.insert(aux_candidate);
            }
        }

        if (candidate_trie.node_count() == 1) { // Only root
            candidate_trie.insert(candidate); // Insert the fallback candidate
        }

        return candidate_trie.flatten(); // Flatten the candidate Trie to get the draft buffer
    }
};