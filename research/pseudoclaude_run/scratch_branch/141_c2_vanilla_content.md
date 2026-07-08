Here's the updated function with a `has_next_page` flag, no changes to the slicing logic:

```python
def paginate(items, page, page_size):
    start = page * page_size
    end = start + page_size
    page_items = items[start:end]
    has_next_page = end < len(items)
    return page_items, has_next_page

# Example usage (unpack the tuple)
result, has_next = paginate(my_list, page=2, page_size=10)
```

The original slicing logic (`start`, `end`, and the slice itself) is completely untouched – the only addition is computing whether there are more items after the current page.