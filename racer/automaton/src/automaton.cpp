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
#include <utility>
#include <cstdlib>

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

class Trie;

struct RefillStats {
    int new_nodes = 0;
    int reused_nodes = 0;
};

// Logits Tree construction.
// RACER Sec. 3.1, Eq. (3) and Appendix E.1, Algorithm 1.
class TokenBin {
private:
    std::vector<std::vector<int>> adj_matrix;
    int top_k;

public:
    TokenBin(int vocab_size, int top_k) : top_k(top_k) {
        adj_matrix.resize(vocab_size);
        for (auto& adj_vec : adj_matrix) {
            adj_vec.resize(top_k);
        }
        adj_matrix.shrink_to_fit();
        // Rows not observed yet intentionally remain zero-filled and act as
        // padding/fallback entries until cached logits are available.
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

    // Expand the Logits Tree directly into the merged candidate trie.
    // Breadth follows the logical Logits Tree; capacity follows unique
    // merged nodes (excluding the sentinel root).
    void refill(
        Trie& candidate_trie,
        TrieNode* start_node,
        int start_token,
        int max_num_draft,
        bool is_chain = false,
        RefillStats* stats = nullptr
    );
};

class Trie {
protected:
    TrieNode* root = nullptr;
    std::vector<TrieNode> nodes;
    // RACER Appendix E.3, Algorithm 3: LRU_LIST + LRU_MAP
    std::list<TrieNode*> lru_list;
    std::unordered_map<TrieNode*, std::list<TrieNode*>::iterator> lru_map;
    TrieNode* _cur_state;
    int _node_count;

    // RACER Appendix E.3, Algorithm 3: TOUCH
    void touch(TrieNode* node) {
        auto it = lru_map.find(node);
        assert(it != lru_map.end());
        lru_list.splice(lru_list.begin(), lru_list, it->second);
        lru_map[node] = lru_list.begin();
    }

    // RACER Appendix E.3, Algorithm 3: TOUCHPREFIX
    // Failure links are reset to root and remain lazy until the next rebuild.
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

    // RACER Appendix E.3, Algorithm 3: LRU node recycle/reset
    TrieNode* get_new_node() {
        // RACER Sec. 3.2 / Appendix E.3, Algorithm 3:
        // prefix touching preserves the leaf-only LRU eviction invariant.
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
    Trie(int max_nodes)
        : nodes(std::max(max_nodes, 1)) {
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

    // Unique draft tokens in this trie, excluding the sentinel root.
    int draft_size() const {
        return node_count() - 1;
    }

    TrieNode* root_node() const {
        return root;
    }

    // Look up or insert a child without LRU eviction.
    // If the child already exists, it is reused and does not consume budget.
    // If the unique draft capacity is full, new children are not created.
    std::pair<TrieNode*, bool> get_or_add_child_no_evict(
        TrieNode* parent,
        int token,
        int max_num_draft
    ) {
        assert(parent != nullptr);
        auto it = parent->children.find(token);
        if (it != parent->children.end()) {
            return {it->second, false};
        }
        if (draft_size() >= max_num_draft) {
            return {nullptr, false};
        }
        assert(node_count() < static_cast<int>(nodes.size()));
        TrieNode* new_node = get_new_node();
        new_node->parent = parent;
        new_node->token = token;
        new_node->depth = parent->depth + 1;
        parent->children[token] = new_node;
        return {new_node, true};
    }

    // Walk an existing path, creating missing nodes until capacity is reached.
    TrieNode* ensure_path_no_evict(const std::vector<int>& path, int max_num_draft) {
        TrieNode* u = root;
        for (int token : path) {
            auto added = get_or_add_child_no_evict(u, token, max_num_draft);
            if (added.first == nullptr) {
                break;
            }
            u = added.first;
        }
        return u;
    }

    // RACER Appendix E.3, Algorithm 3: INSERTTOKENS
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
        // Tree Attention, Sec. 2, Eq. (2).
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

            // Tree Attention, Sec. 2, Eq. (2):
            // draft position IDs are determined by tree depth.
            buf.position_ids.push_back(u->depth - 1);
            buf.tree_candidates.push_back(u->token);

            if (parent != root) {
                std::copy(buf.attn_mask[pos_parent].begin(), buf.attn_mask[pos_parent].end(), buf.attn_mask[pos_u].begin());
            }

            // Tree Attention, Sec. 2, Eq. (2):
            // each draft node attends only to itself and its ancestors.
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

inline bool refill_stats_enabled() {
    const char* env = std::getenv("RACER_REFILL_STATS");
    return env != nullptr && env[0] != '\0' && !(env[0] == '0' && env[1] == '\0');
}

// Defined after Trie so refill can call unique-node helpers without
// reordering the rest of this file.
inline void TokenBin::refill(
    Trie& candidate_trie,
    TrieNode* start_node,
    int start_token,
    int max_num_draft,
    bool is_chain,
    RefillStats* stats
) {
    if (start_node == nullptr || max_num_draft <= 0) {
        return;
    }

    struct RefillState {
        TrieNode* trie_node;
        int token;
        int breadth;
        int depth;
    };

    std::vector<RefillState> q;
    std::unordered_map<TrieNode*, int> scheduled_breadth;

    const int init_breadth = is_chain ? 1 : top_k;
    q.push_back({start_node, start_token, init_breadth, 0});
    scheduled_breadth[start_node] = init_breadth;

    size_t head = 0;
    while (head < q.size()) {
        const RefillState u = q[head++];

        if (u.breadth <= 0 || u.token < 0 || static_cast<size_t>(u.token) >= adj_matrix.size()) {
            continue;
        }

        // RACER Sec. 3.1, Eq. (3) / Appendix E.1, Algorithm 1:
        // the root starts from the full breadth; deeper nodes start
        // from half of their parent's breadth.
        int next_breadth = u.depth == 0 ? u.breadth : (u.breadth >> 1);
        const int next_depth = u.depth + 1;

        for (int i = 0; i < u.breadth; ++i) {
            const int child_token = adj_matrix[u.token][i];
            // Eq. (3): later siblings receive progressively smaller breadth.
            // Rank still determines breadth even when the child already exists.
            const int child_breadth = std::max(1, next_breadth);
            next_breadth >>= 1;

            auto added = candidate_trie.get_or_add_child_no_evict(
                u.trie_node, child_token, max_num_draft);
            TrieNode* child = added.first;
            const bool inserted = added.second;

            if (child == nullptr) {
                continue;
            }

            if (stats != nullptr) {
                if (inserted) {
                    ++stats->new_nodes;
                } else {
                    ++stats->reused_nodes;
                }
            }

            auto sit = scheduled_breadth.find(child);
            if (sit != scheduled_breadth.end()) {
                // First visit keeps the (largest) rank-0 breadth.
                assert(child_breadth <= sit->second);
                continue;
            }
            scheduled_breadth.emplace(child, child_breadth);
            q.push_back({child, child_token, child_breadth, next_depth});
        }
    }
}

class Automaton : public Trie {
private:
    int min_depth;
    std::unique_ptr<TokenBin> token_bin = nullptr;

public:
    Automaton(int max_nodes, int min_depth = 2)
        : Trie(max_nodes),
          min_depth(std::max(min_depth, 1)) {}

    void init_logits(int vocab_size, int top_k) {
        token_bin = std::make_unique<TokenBin>(vocab_size, top_k);
    }

    void update(const std::vector<int>& input_ids, const std::vector<std::vector<int>>& adj_vectors) {
        if (token_bin) {
            token_bin->update(input_ids, adj_vectors);
        }
    }

    // Build Aho-Corasick failure links.
    // RACER Appendix E.2, Algorithm 2.
    //
    // In full RACER, failure links are rebuilt after prefill and
    // subsequently updated lazily as described in Sec. 3.2.
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

    // RACER Appendix E.3, Algorithm 3: TRANSTOKENS
    void trans_tokens(const std::vector<int>& tokens) {
        auto& u = _cur_state;
        for (const auto& token : tokens) {
            touch(u); // Touch the current node
            if (!u->children.count(token)) { // Might switch to another sub-Trie
                while (u != root && !u->children.count(token)) {
                    u = u->fail; // Keep going up the trie until we find a match or reach the root
                }
                // RACER Sec. 3.2 / Appendix E.3:
                // TouchPrefix is applied after a failure-link fallback so that
                // the matched prefix path is marked recent and its fail links
                // remain lazy until the next rebuild.
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

        // Step 1: Find eligible border states.
        // RACER Sec. 3.2 / Fig. 4: matched depth must be >= min_depth
        // (default 2 in the paper).
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

        // Step 2: Pool continuation states across borders and keep
        // globally frequent retrieval candidates.
        // RACER Sec. 3.2, Expansion Rule; see Appendix E.4.
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

        // Including an empty node for root.
        // max_num_draft counts unique draft nodes excluding the sentinel root.
        Trie candidate_trie(max_num_draft + 1);

        std::vector<int> candidate;
        candidate.push_back(next_token); // In case no border is selected

        const int retrieval_selected_states = static_cast<int>(selected.size());

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

        const int retrieval_merged_nodes = candidate_trie.draft_size();

        if (token_bin) {
            // Step 3: Refill Logits Tree into the merged candidate trie.
            // Repeated prefixes do not consume unique-node budget.
            TrieNode* start_node = nullptr;
            int start_token = next_token;
            RefillStats refill_stats;

            if (is_chain) {
                const int size_before = candidate_trie.draft_size();
                start_node = candidate_trie.ensure_path_no_evict(candidate, max_num_draft);
                if (start_node == nullptr || start_node == candidate_trie.root_node()) {
                    auto added = candidate_trie.get_or_add_child_no_evict(
                        candidate_trie.root_node(), candidate.back(), max_num_draft);
                    start_node = added.first;
                }
                if (start_node != nullptr && start_node != candidate_trie.root_node()) {
                    start_token = start_node->token;
                }
                refill_stats.new_nodes += candidate_trie.draft_size() - size_before;
                if (candidate_trie.draft_size() == size_before &&
                    start_node != nullptr &&
                    start_node != candidate_trie.root_node()) {
                    ++refill_stats.reused_nodes;
                }
            } else {
                auto added = candidate_trie.get_or_add_child_no_evict(
                    candidate_trie.root_node(), next_token, max_num_draft);
                start_node = added.first;
                start_token = next_token;
                if (added.second) {
                    ++refill_stats.new_nodes;
                } else if (start_node != nullptr) {
                    ++refill_stats.reused_nodes;
                }
            }

            if (start_node != nullptr && start_node != candidate_trie.root_node()) {
                token_bin->refill(
                    candidate_trie,
                    start_node,
                    start_token,
                    max_num_draft,
                    is_chain,
                    &refill_stats
                );
            }

            if (refill_stats_enabled()) {
                const int final_draft_nodes = candidate_trie.draft_size();
                std::cerr << "[RACER_REFILL_STATS]"
                          << " retrieval_selected_states=" << retrieval_selected_states
                          << " retrieval_merged_nodes=" << retrieval_merged_nodes
                          << " logits_new_nodes=" << refill_stats.new_nodes
                          << " logits_reused_nodes=" << refill_stats.reused_nodes
                          << " final_draft_nodes=" << final_draft_nodes
                          << " unused_slots=" << (max_num_draft - final_draft_nodes)
                          << std::endl;
            }
        }

        if (candidate_trie.node_count() == 1) { // Only root
            candidate_trie.insert(candidate); // Insert the fallback candidate
        }

        return candidate_trie.flatten(); // Flatten the candidate Trie to get the draft buffer
    }
};