# SAE Reading Time

## SparseRT

SparseRT investigates what GPT-2 small is *not computing* that humans are; however, by applying Sparse Autoencoders (SAEs) across all 12 layers of the model and identifying which internal features are suppressed on words that humans find cognitively costly to read.

### Research Question

> Which SAE features are systematically zeroed out on high reading-time words, and what do those features represent linguistically?

Rather than treating surprisal as a black box, we decompose GPT-2's internal representations layer-by-layer into sparse, interpretable features and ask: does the model fail to activate the same features a human reader would rely on?