# pandas-stubs Type Checking Philosophy

The goal of the pandas-stubs project is to provide type stubs for the public API
that represent the recommended ways of using pandas.  This is opposed to the
philosophy within the pandas source, as described [here](https://pandas.pydata.org/docs/development/contributing_codebase.html?highlight=typing#type-hints), which
is to assist with the development of the pandas source code to ensure type safety within
that source.

Due to the methodology used by Microsoft to develop the original stubs, there are internal
classes, methods and functions that are annotated within the pandas-stubs project
that are incorrect with respect to the pandas source, but that have no effect on type
checking user code that calls the public API.

## Use of Generic Types

There are other differences that are extensions of the pandas API to assist in type
checking.  Two key examples are that `Series` and `Interval` are typed as generic types.

### Series are Generic

`Series` is declared as `Series[S1]` where `S1` is a `TypeVar` consisting of types normally
used within series, if that type can be inferred.  Consider the following example
that compares the values in a `Series` to an integer.

```python
s = pd.Series([1, 2, 3])
lt = s < 3
```

In the pandas source, `lt` is a `Series` with a `dtype` of `bool`.  In the pandas-stubs,
the type of `lt` is `Series[bool]`.  This allows further type checking to occur in other
pandas methods.  Note that in the above example, `s` is typed as `Series[Any]` because
its type cannot be statically inferred.

This also allows type checking for operations on series that contain date/time data.  Consider
the following example that creates two series of datetimes with corresponding arithmetic.

```python
s1 = pd.Series(pd.to_datetime(["2022-05-01", "2022-06-01"]))
reveal_type(s1)
s2 = pd.Series(pd.to_datetime(["2022-05-15", "2022-06-15"]))
reveal_type(s2)
td = s1 - s2
reveal_type(td)
ssum = s1 + s2
reveal_type(ssum)
```

The above code (without the `reveal_type()` statements) will raise an `Exception` on the computation of `ssum` because it is
inappropriate to add two series containing `Timestamp` values.  The types will be
revealed as follows:

```text
ttest.py:4: note: Revealed type is "pandas.core.series.TimestampSeries"
ttest.py:6: note: Revealed type is "pandas.core.series.TimestampSeries"
ttest.py:8: note: Revealed type is "pandas.core.series.TimedeltaSeries"
ttest.py:10: note: Revealed type is "builtins.Exception"
```

The type `TimestampSeries` is the result of creating a series from `pd.to_datetime()`, while
the type `TimedeltaSeries` is the result of subtracting two `TimestampSeries` as well as
the result of `pd.to_timedelta()`.

### Interval is Generic

A pandas `Interval` can be a time interval, an interval of integers, or an interval of
time, represented as having endpoints with the `Timestamp` class.  pandas-stubs tracks
the type of an `Interval`, based on the arguments passed to the `Interval` constructor.
This allows detecting inappropriate operations, such as adding an integer to an
interval of `Timestamp`s.

## Testing the Type Stubs

A set of (most likely incomplete) tests for testing the type stubs is in the pandas-stubs
repository in the `tests` directory.  The tests are used with `mypy` and `pyright` to
validate correct typing, and also with `pytest` to validate that the provided code
actually executes.  The recent decision for Python 3.11 to include `assert_type()`,
which is supported by `typing_extensions` version 4.2 and beyond makes it easier
to test to validate the return types of functions and methods.  Future work
is intended to expand the use of `assert_type()` in the test code.

## Narrow vs. Wide Arguments

A consideration in creating stubs is to make the set of type annotations for
function arguments "just right", i.e.,
not too narrow and not too wide.  A type annotation to an argument to a function or
method is too narrow if it disallows valid arguments.  A type annotation to
an argument to a function or method is too wide if
it allows invalid arguments.  Testing for type annotations that are too narrow is rather
straightforward.  It is easy to create an example for which the type checker indicates
the argument is incorrect, and add it to the set of tests in the pandas-stubs
repository after fixing the appropriate stub.  However, testing for when type annotations
are too wide is a bit more complicated.
In this case, the test will fail when using `pytest`, but it is also desirable to
have type checkers report errors for code that is expected to fail type checking.

Here is an example that illustrates this concept, from `tests/test_interval.py`:

```python
    i1 = pd.Interval(
        pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-03"), closed="both"
    )
    if TYPE_CHECKING_INVALID_USAGE:
        i1 + pd.Timestamp("2000-03-03")  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]

```

In this particular example, the stubs consider that `i1` will have the type
`pd.Interval[pd.Timestamp]`.  It is incorrect code to add a `Timestamp` to a
time-based interval.  Without the `if TYPE_CHECKING_INVALID_USAGE` construct, the
code would fail at runtime.  Further, type checkers should report an error for this
incorrect code.  By placing the `# type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]`
on the line, type checkers are told to ignore the type error.  To ensure that the
pandas-stubs annotations are not too wide (allow adding a `Timestamp` to a
time-based interval), mypy and pyright are configured to report unused ignore
statements.

## Using ignore comments

Type checkers report errors

- when writing negative tests to reject invalid behavior (inside a
  `TYPE_CHECKING_INVALID_USAGE` block),
- when writing `overload`s that return incompatible return values, or
- when type checkers have bugs themselves.

Since mypy and pyright behave slightly differently, we use separate ignore comments
for them.

- If mypy reports an error, please use `# type: ignore[<error code>]`
- If pyright reports an error, please use `# pyright: ignore[<error code>]`

If mypy and pyright report errors, for example, inside a `TYPE_CHECKING_INVALID_USAGE`
block, please ensure that the comment for mypy comes first:
`# type: ignore[<error code>] # pyright: ignore[<error code>]`.
