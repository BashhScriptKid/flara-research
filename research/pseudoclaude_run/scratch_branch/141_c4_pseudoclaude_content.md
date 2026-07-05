```python
from typing import Any

def add_item(item: Any, cart: list[Any] = []) -> list[Any]:
    """Add an item to a cart list and return the updated list.

    Args:
        item: The item to add to the cart.
        cart: An optional list representing the cart. Defaults to an empty list.
            **Note:** Because Python evaluates default arguments only once, the
            same default list is reused on every call that does not provide an
            explicit cart, which can lead to shared state between calls.

    Returns:
        The cart list after appending the item.
    """
    cart.append(item)
    return cart
```