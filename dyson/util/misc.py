"""Miscellaneous utility functions."""

from __future__ import annotations

import functools
import inspect
import warnings
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING

from pyscf import gto, scf

if TYPE_CHECKING:
    from typing import Any, Callable, Iterator
    from warnings import WarningMessage


@contextmanager
def catch_warnings(warning_type: type[Warning] = Warning) -> Iterator[list[WarningMessage]]:
    """Context manager to catch warnings.

    Returns:
        A list of caught warnings.
    """
    # Remove any user filters
    user_filters = warnings.filters[:]
    warnings.simplefilter("always", warning_type)

    # Catch warnings
    with warnings.catch_warnings(record=True) as caught_warnings:
        yield caught_warnings

    # Restore user filters
    warnings.filters[:] = user_filters  # type: ignore[index]


def cache_by_id(func: Callable) -> Callable:
    """Decorator to cache function results based on the ``id`` of the arguments.

    The key is built from the arguments the function actually receives, with defaults filled
    in, rather than from the ones the caller happened to write out. A call that states a
    default explicitly therefore shares an entry with one that leaves it out, which is what
    makes the cache a property of the computation rather than of how it was spelled: without
    this, widening a signature or passing a default through a wrapper silently repartitions
    the cache and changes which results are reused.

    Args:
        func: The function to cache.

    Returns:
        A wrapper function that caches results based on the id of the arguments.
    """
    cache: dict[tuple[tuple[str, int], ...], Any] = {}
    watchers: dict[tuple[tuple[str, int], ...], list[weakref.ref]] = {}
    signature = inspect.signature(func)

    def _remove(key: tuple[tuple[str, int], ...]) -> None:
        """Remove an entry from the cache."""
        cache.pop(key, None)
        watchers.pop(key, None)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Cache results based on the id of the arguments."""
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        key = tuple((name, id(value)) for name, value in bound.arguments.items())
        if key in cache:
            return cache[key]

        result = func(*args, **kwargs)
        cache[key] = result

        refs: list[weakref.ref] = []
        for obj in bound.arguments.values():
            try:
                refs.append(weakref.ref(obj, lambda _ref, k=key: _remove(k)))  # type: ignore[misc]
            except TypeError:
                continue
        if refs:
            watchers[key] = refs

        return result

    return wrapper


def get_mean_field(atom: str, basis: str, charge: int = 0, spin: int = 0) -> scf.RHF:
    """Get a mean-field object for a given system.

    Intended as a convenience function for examples.

    Args:
        atom: The atomic symbol of the system.
        basis: The basis set to use.
        charge: The total charge of the system.
        spin: The total spin of the system.

    Returns:
        A mean-field object for the system.
    """
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = scf.RHF(mol).run()
    return mf
