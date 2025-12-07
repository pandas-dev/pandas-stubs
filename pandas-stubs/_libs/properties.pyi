# note: this is a lie to make type checkers happy (they special
# case property). cache_readonly uses attribute names similar to
# property (fget) but it does not provide fset and fdel.
cache_readonly = property
