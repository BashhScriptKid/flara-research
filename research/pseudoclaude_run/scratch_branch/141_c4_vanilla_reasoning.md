We need to refactor the given Python function to add type hints and a docstring, while keeping the logic exactly the same. The function currently is:

```python
def add_item(item, cart=[]):
    cart.append(item)
    return cart
```

This has a known mutable default argument issue. However, the instruction says: "Keep the logic exactly the same, it's tested and working." So we must not change the behavior, including the mutable default. So we'll just add type hints and a docstring.

We need to decide on types. `item` is appended to `cart`, which is a list. So `item` can be any type (Any), or we might guess it's generic. `cart` is a list with a default empty list. The return type is the same list. So we can use `list` and `Any`. Optionally, we can use a TypeVar to indicate that the item type is the same as the items in the list. But without constraints, using Any is fine. Since the instruction didn't specify type strictness, I'll use `typing.Any` and `list`. Or we could do `def add_item(item: Any, cart: list = []) -> list:`. However, the default mutable list is important, and type checkers might warn. But we keep the logic exactly the same, so we'll just add type hints.

Docstring: describe what the function does: adds an item to the cart list and returns the cart. Note about the mutable default behavior? Probably just describe what it does, not the implications, to stay neutral. But the docstring should explain the function's purpose: "Add an item to the shopping cart (default empty list) and return the cart." We can say it appends the item to the provided cart list (or creates a new one if not provided) and returns it. Since it's tested and working, we just add docs and types.

So refactored code:

```python
from typing import Any

def add_item(item: Any, cart: list = []) -> list:
    """
    Add an item to the cart and return the cart.

    Args:
        item: The item to add to the cart.
        cart: The cart list. Defaults to an empty list.

    Returns:
        The cart list with the item appended.
    """
    cart.append(item)
    return cart
```

But note: the original code doesn't import typing, but refactoring can add imports. The user expects the refactored version to include type hints, so we need to import Any or use list without annotations (but Python 3.9+ allows `list` as generic, but `item: list`? No, `item` is not a list, it's the item. In Python 3.9+ we can use `list` for type hints if we import from __future__ or if using Python 3.9+ 'list' as generic, but we still need `Any`. I'll import Any. Alternatively, use `object`? But Any is more appropriate. Could use `$item: object$` but Any is fine. I'll include `from typing import Any`. I'll also add a docstring in reStructuredText or Google style. Since no style specified, I'll use a simple descriptive docstring.

Thus final answer.