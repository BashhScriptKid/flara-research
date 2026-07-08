The simplest way to add a `has_next_page` flag without touching the slicing logic is to return a tuple of `(sliced_items, has_next_page)`. The slicing remains identical, and the flag is computed by checking whether the `end` index falls before the end of the list.

```python
def paginate(items, page, page_size):
    start = page * page_size
    end = start + page_size
    sliced = items[start:end]
    has_next_page = end < len(items)
    return sliced, has_next_page
```

The original call would now return a tuple, so adjust the usage:

```python
result_items, has_next_page = paginate(my_list, page=2, page_size=10)
```

If you need backward compatibility and can’t change the call sites, you could wrap the return in an object (e.g. a namedtuple or a custom class) that exposes both `.items` and `.has_next_page` attributes—but the above tuple approach is the simplest change while preserving the existing slicing logic.