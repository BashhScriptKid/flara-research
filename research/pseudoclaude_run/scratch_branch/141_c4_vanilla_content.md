```python
from typing import Any

def add_item(item: Any, cart: list = []) -> list:
    """
    Add an item to the cart and return the cart.

    If no cart is provided, a new empty list is used.
    
    Args:
        item: The item to be added to the cart.
        cart: The cart list. Defaults to an empty list.

    Returns:
        The cart list after the item has been appended.
    """
    cart.append(item)
    return cart
```