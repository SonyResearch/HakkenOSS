# Simple Querying Package

This implements the simple querying module.
It assumes that there is only one unknown link (i.e. `(?, r, o)`, `(s, ?, o)` or `(s, r, ?)`) along with multiple known (existing) links.
The simple querying module is not capable of processing complex queries (i.e. DNF queries connected with AND/OR, negated conditions, etc.),
but in exchange it can process non-DNF conditions on existing links (e.g. `(A AND (B OR C) AND D)`).
